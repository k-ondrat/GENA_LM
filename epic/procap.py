"""Шаг A: ProCapNet-подобная модель с нуля для одного вида (профиль по основаниям обеих цепей + счёт).

    python epic/procap.py train   A [--tag pc] [--steps 60000] [--eval-every 4000]
    python epic/procap.py predict A --tag pc [--intervals whitelist.test.bed.gz] [--contig-list F] [--out DIR]

Вход 2114 п.н. one-hot, выход — центральные 1000 п.н. × 2 цепи (каналы: '+', '-' в геномных координатах).
Loss = мультиномиальный NLL профиля (softmax по размеченным позициям тайла, обе цепи; вес тайла min(N, count_cap)/N)
       + alpha·MSE(log1p N).
Предсказание: lambda = expm1(c) · softmax(профиль по размеченным/скоримым позициям).

valid-оценка каждые --eval-every шагов — по ВСЕМ размеченным позициям valid-контигов (как в скорере):
  AUPRC, Spearman по позитивам (скор = log1p(lambda)/max, 5 знаков), Spearman счёта по тайлам,
  NLL на рид: M (модель), U (равномерно по тайлу), R (одна реплика предсказывает другую),
  G = (U - M) / (U - R) — доля закрытого разрыва до уровня реплик.
Выход: epic_train/<A>/procap/<tag>.{tsv,png}, step*.pt, best.pt (по AUPRC); predict — npz как predict.py.
"""
import argparse, glob, gzip, math, os, time
import numpy as np, pandas as pd, pysam, torch
import torch.nn as nn, torch.nn.functional as F
from collections import defaultdict
from scipy.stats import spearmanr

H = os.environ["GENALM_HOME"]
IN, OUT = 2114, 1000
FL, PAD = (IN - OUT) // 2, 2000          # PAD >= OUT + FL
LUT = np.full(256, 4, np.uint8)
for i, ch in enumerate(b"ACGT"):
    LUT[ch] = LUT[ch + 32] = i


# ---------------- данные ----------------
def load_vals(A, name):
    """Одно число на позицию template.train (int32), с кэшем .npy."""
    fp, cache = f"{H}/epic_data/{A}/for_participants", f"{H}/epic_train/{A}/procap/{name}.npy"
    if not os.path.exists(cache):
        f = f"{fp}/csRNA-train.txt.gz" if name == "total" else glob.glob(f"{fp}/csRNA-*-{name}-train.txt.gz")[0]
        np.save(cache, pd.read_csv(f, header=None, dtype=np.int32).iloc[:, 0].to_numpy())
    return np.load(cache, mmap_mode="r")


def pack(A, contigs, vals=(), mask_from=None):
    """Склейка контигов с N-паддингом: seq (Lt,) uint8, массивы (2, Lt) int32 на каждое vals (-1 = нет разметки),
    mask (2, Lt) bool (скоримые позиции), offsets {контиг: начало}."""
    fa = pysam.FastaFile(f"{H}/epic_data/{A}/genome/genome.fa")
    off, L = {}, PAD
    for c in contigs:
        off[c] = L
        L += fa.get_reference_length(c) + PAD
    seq = np.full(L, 4, np.uint8)
    for c in contigs:
        seq[off[c]:off[c] + fa.get_reference_length(c)] = LUT[np.frombuffer(fa.fetch(c).encode(), np.uint8)]
    ys = [np.full((2, L), -1, np.int32) for _ in vals]
    mask = np.zeros((2, L), bool)
    t = mask_from
    if t is not None:
        t = t[t.c.isin(off)]
        for c, s, e, st, o in zip(t.c, t.s, t.e, t.st, t.off):
            g, si = off[c] + s, (0 if st == "+" else 1)
            mask[si, g:g + e - s] = True
            for y, v in zip(ys, vals):
                y[si, g:g + e - s] = v[o:o + e - s]
    return seq, ys, mask, off


def template(A, name="template.train.bed.gz"):
    t = pd.read_csv(f"{H}/epic_data/{A}/for_participants/{name}", sep="\t", header=None,
                    usecols=[0, 1, 2, 5], names=["c", "s", "e", "st"])
    t["off"] = np.r_[0, np.cumsum((t.e - t.s).to_numpy())[:-1]]
    return t


def AC(dev):
    return torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")


def onehot(x):  # (B, IN) uint8 -> (B, 4, IN) float, N = нули
    return F.one_hot(x.long(), 5)[..., :4].transpose(1, 2).float()


def rc(x, prof):  # reverse complement входа и выхода (каналы цепей меняются местами)
    return x.flip(-1)[:, [3, 2, 1, 0]], (None if prof is None else prof.flip(-1)[:, [1, 0]])


# ---------------- модель ----------------
class ProCap(nn.Module):
    def __init__(self, f=512, n=8):
        super().__init__()
        self.c0 = nn.Conv1d(4, f, 21, padding=10)
        self.dil = nn.ModuleList(nn.Conv1d(f, f, 3, padding=2 ** i, dilation=2 ** i) for i in range(1, n + 1))
        self.prof = nn.Conv1d(f, 2, 75, padding=37)
        self.cnt = nn.Linear(f, 1)

    def forward(self, x):  # (B, 4, IN) -> профиль (B, 2, OUT), log1p счёта (B,)
        h = F.relu(self.c0(x))
        for c in self.dil:
            h = h + F.relu(c(h))
        p = self.prof(h)[..., FL:FL + OUT]
        return p, self.cnt(h[..., FL:FL + OUT].mean(-1)).squeeze(-1)


def masked_logp(prof, m):  # log softmax по размеченным позициям тайла (обе цепи); вне маски 0
    lp = prof.float().masked_fill(~m, -1e30).flatten(1).log_softmax(-1).view_as(prof)
    return lp.masked_fill(~m, 0)


def lam(prof, c, m):  # ожидаемый счёт на позицию
    return torch.expm1(c.float().clamp(max=20)).clamp(min=0)[:, None, None] * masked_logp(prof, m).exp() * m


# ---------------- метрики ----------------
def ap_ties(s, pos):  # AP с ничьими как у скорера (s, pos — torch на одном устройстве)
    s, o = torch.sort(s, descending=True)
    p = pos[o].double()
    tp, n = p.cumsum(0), torch.arange(1, len(p) + 1, device=p.device, dtype=torch.float64)
    last = torch.ones_like(s, dtype=torch.bool)
    last[:-1] = s[1:] != s[:-1]
    tp, n = tp[last], n[last]
    return float(((tp / n) * torch.diff(tp, prepend=tp.new_zeros(1))).sum() / p.sum())


def tile_nll(y, lp):  # сумма -y·log p по тайлу
    return -(y * lp).sum((1, 2))


class Valid:
    def __init__(self, A, contigs, t, dev):
        tot, r1 = load_vals(A, "total"), load_vals(A, "r1")
        seq, (y, y1), mask, off = pack(A, contigs, (tot, r1), t)
        st, offs = [], sorted(off.values())
        for o, nxt in zip(offs, offs[1:] + [len(seq)]):
            lab = np.flatnonzero(mask[:, o:nxt].any(0))
            if len(lab):
                st += list(range(o + lab[0], o + lab[-1] + 1, OUT))
        self.st = torch.tensor(st, device=dev)
        self.seq, self.y, self.y1 = (torch.from_numpy(a).to(dev) for a in (seq, y, y1))
        self.m = torch.from_numpy(mask).to(dev)
        self.R = self.rep_nll()

    def tiles(self, st):
        ar_in, ar_out = torch.arange(IN, device=st.device), torch.arange(OUT, device=st.device)
        x = onehot(self.seq[st[:, None] - FL + ar_in])
        g = lambda a: a[:, st[:, None] + ar_out].permute(1, 0, 2)
        return x, g(self.m), g(self.y).clamp(min=0).float(), g(self.y1).clamp(min=0).float()

    def rep_nll(self, B=2048):
        """NLL на рид, когда одна реплика (со сглаживанием eps) предсказывает другую; лучший eps."""
        num, den = defaultdict(float), 0.0
        for i in range(0, len(self.st), B):
            _, m, y, y1 = self.tiles(self.st[i:i + B])
            y2 = y - y1
            den += float(y1.sum() + y2.sum())
            for eps in (0.01, 0.1, 0.5, 1.0):
                for a, b in ((y1, y2), (y2, y1)):
                    p = (a + eps) * m
                    lp = torch.log(p / p.sum((1, 2), keepdim=True).clamp(min=1e-30)).masked_fill(~m, 0)
                    num[eps] += float(tile_nll(b, lp).sum())
        return min(num.values()) / den

    @torch.no_grad()
    def evaluate(self, model, B=512):
        model.eval()
        S, Y, Nt, Ct, Mn, Un, Nr = [], [], [], [], 0.0, 0.0, 0.0
        for i in range(0, len(self.st), B):
            x, m, y, _ = self.tiles(self.st[i:i + B])
            with AC(x.device):
                prof, c = model(x)
            S.append(lam(prof, c, m)[m]); Y.append(y[m])
            k, N = m.sum((1, 2)), y.sum((1, 2))
            ok = k >= OUT                                   # тайл размечен >= 50% (из 2·OUT позиций)
            Nt.append(N[ok]); Ct.append(c.float()[ok])
            w = ok & (N > 0)
            Mn += float(tile_nll(y, masked_logp(prof, m))[w].sum())
            Un += float((N * torch.log(k.float().clamp(min=1)))[w].sum())
            Nr += float(N[w].sum())
        model.train()
        s, y = torch.log1p(torch.cat(S)), torch.cat(Y)
        s = torch.round(s / s.max().clamp(min=1e-30) * 1e5) / 1e5
        pos = y > 0
        M, U = Mn / Nr, Un / Nr
        return dict(auprc=ap_ties(s, pos), spearman=spearmanr(y[pos].cpu().numpy(), s[pos].float().cpu().numpy())[0],
                    count_spearman=spearmanr(torch.cat(Nt).cpu().numpy(), torch.cat(Ct).cpu().numpy())[0],
                    M=M, U=U, R=self.R, G=(U - M) / (U - self.R))


# ---------------- обучение ----------------
def train(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(a.seed)
    D = f"{H}/epic_train/{a.assembly}/procap"
    os.makedirs(D, exist_ok=True)
    t = template(a.assembly)
    valid = set(open(f"{H}/epic_train/{a.assembly}/intervals_full/valid_contigs.txt").read().split())
    tr_c = [c for c in t.c.unique() if c not in valid]
    print(f"train-контигов {len(tr_c)}, valid {len(valid)}; загрузка...", flush=True)
    seq, (y,), _, _ = pack(a.assembly, tr_c, (load_vals(a.assembly, "total"),), t)
    seq, y = torch.from_numpy(seq).to(dev), torch.from_numpy(y).to(dev)
    V = Valid(a.assembly, sorted(valid), t, dev)
    pos_idx = torch.nonzero((y > 0).any(0)).squeeze(1)
    Lt = y.shape[1]
    ar_in, ar_out = torch.arange(IN, device=dev), torch.arange(OUT, device=dev)
    print(f"train: {Lt / 1e6:.0f} млн п.н., позитивных позиций {len(pos_idx):,} | valid: тайлов {len(V.st):,}, "
          f"R = {V.R:.4f}", flush=True)

    def batch(B):
        nb = B - int(B * a.bg_frac)
        sp = pos_idx[torch.randint(len(pos_idx), (nb,), device=dev)] - torch.randint(OUT, (nb,), device=dev)
        sb = torch.randint(FL, Lt - OUT - FL, (4 * (B - nb),), device=dev)
        st = torch.cat([sp.clamp(FL, Lt - OUT - FL), sb])
        yy = y[:, st[:, None] + ar_out].permute(1, 0, 2)
        ok = (yy >= 0).sum((1, 2)) >= OUT                   # >= 50% размечено
        ok[nb:] &= torch.cumsum(ok[nb:].int(), 0) <= B - nb
        st, yy = st[ok], yy[ok]
        x = onehot(seq[st[:, None] - FL + ar_in])
        r = torch.rand(len(st), device=dev) < 0.5
        x[r], yy[r] = x[r].flip(-1)[:, [3, 2, 1, 0]], yy[r].flip(-1)[:, [1, 0]]
        return x, yy >= 0, yy.clamp(min=0).float()

    _, m0, y0 = batch(8192)
    N0 = y0.sum((1, 2))
    alpha = a.alpha or max(1.0, float(N0[N0 > 0].clamp(max=a.count_cap).median()) / 2)
    print(f"alpha = {alpha:.2f}", flush=True)

    model = ProCap(a.filters, a.layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / a.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    log, best, acc, t0 = f"{D}/{a.tag}.tsv", -1, defaultdict(float), time.time()
    rows = []
    for step in range(1, a.steps + 1):
        x, m, yy = batch(a.batch)
        with AC(dev):
            prof, c = model(x)
        N = yy.sum((1, 2))
        w = N.clamp(max=a.count_cap) / N.clamp(min=1)          # тайл весит не больше count_cap ридов
        nll = (tile_nll(yy, masked_logp(prof, m)) * w).mean()
        mse = F.mse_loss(c.float(), torch.log1p(N))
        loss = nll + alpha * mse
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        acc["nll"] += nll.item(); acc["mse"] += mse.item(); acc["n"] += 1
        if step % a.eval_every == 0 or step == a.steps:
            r = V.evaluate(model)
            r.update(step=step, train_nll=acc["nll"] / acc["n"], train_mse=acc["mse"] / acc["n"],
                     lr=sched.get_last_lr()[0], min=(time.time() - t0) / 60)
            acc.clear(); rows.append(r)
            pd.DataFrame(rows).to_csv(log, sep="\t", index=False, float_format="%.5g")
            torch.save(model.state_dict(), f"{D}/{a.tag}.step{step}.pt")
            if r["auprc"] > best:
                best = r["auprc"]; torch.save(model.state_dict(), f"{D}/{a.tag}.best.pt")
            print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()), flush=True)
            plot(pd.DataFrame(rows), f"{D}/{a.tag}.png", a.assembly)
    print(f"лучший AUPRC {best:.4f} -> {D}/{a.tag}.best.pt")


def plot(df, out, title):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.5))
    for axx, cols in zip(ax, (["auprc"], ["spearman", "count_spearman"], ["G"], ["train_nll", "M", "U", "R"])):
        for k in cols:
            axx.plot(df.step, df[k], marker=".", label=k)
        axx.legend(); axx.set_xlabel("шаг"); axx.grid(alpha=.3)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


# ---------------- инференс ----------------
@torch.no_grad()
def predict(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = f"{H}/epic_train/{a.assembly}/procap"
    model = ProCap(a.filters, a.layers).to(dev)
    model.load_state_dict(torch.load(a.checkpoint or f"{D}/{a.tag}.best.pt", map_location=dev))
    model.eval()
    out = a.out or f"{H}/predictions/{a.assembly}_{a.tag}"
    os.makedirs(out, exist_ok=True)
    stranded = "template" in a.intervals
    t = pd.read_csv(f"{H}/epic_data/{a.assembly}/for_participants/{a.intervals}", sep="\t", header=None)
    t = pd.DataFrame({"c": t[0], "s": t[1], "e": t[2], "st": t[5] if stranded else "."})
    if a.contig_list:
        t = t[t.c.isin(set(open(a.contig_list).read().split()))]
    fa = pysam.FastaFile(f"{H}/epic_data/{a.assembly}/genome/genome.fa")
    ar_in, ar_out = torch.arange(IN, device=dev), torch.arange(OUT, device=dev)
    for c, g in t.groupby("c", sort=False):
        if all(os.path.exists(f"{out}/{c}{s}.npz") for s in "+-"):
            continue
        lo, hi, L = int(g.s.min()), int(g.e.max()), fa.get_reference_length(c)
        seq = np.full(L + 2 * PAD, 4, np.uint8)
        seq[PAD:PAD + L] = LUT[np.frombuffer(fa.fetch(c).encode(), np.uint8)]
        m = np.zeros((2, L + 2 * PAD), bool)
        for s, e, st in zip(g.s, g.e, g.st):
            for si in ([0] if st == "+" else [1] if st == "-" else [0, 1]):
                m[si, PAD + s:PAD + e] = True
        seq, m = torch.from_numpy(seq).to(dev), torch.from_numpy(m).to(dev)
        track = torch.zeros(2, hi - lo, device=dev)
        st_all = torch.arange(PAD + lo, PAD + hi, OUT, device=dev)
        for i in range(0, len(st_all), a.batch):
            st = st_all[i:i + a.batch]
            x = onehot(seq[st[:, None] - FL + ar_in])
            mm = m[:, st[:, None] + ar_out].permute(1, 0, 2)
            with AC(dev):
                p1, c1 = model(x)
                xr, _ = rc(x, None)
                p2, c2 = model(xr)
            _, p2 = rc(xr, p2)
            v = (lam(p1, c1, mm) + lam(p2, c2, mm)) / 2          # (b, 2, OUT), TTA по цепям
            for j, s0 in enumerate((st - PAD - lo).tolist()):
                n = min(OUT, hi - lo - s0)
                track[:, s0:s0 + n] = v[j, :, :n]
        track = track.cpu().numpy()
        for si, s in enumerate("+-"):
            np.savez(f"{out}/{c}{s}.npz", lo=lo, track=track[si])
        print(f"{c}: {hi - lo:,} п.н.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "predict"])
    p.add_argument("assembly")
    p.add_argument("--tag", default="pc")
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--eval-every", type=int, default=4000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--bg-frac", type=float, default=0.5, help="доля фоновых тайлов в батче")
    p.add_argument("--alpha", type=float, default=0, help="вес MSE счёта; 0 = медиана N позитивных тайлов / 2")
    p.add_argument("--count-cap", type=float, default=100, help="макс. вес тайла в профильном loss, ридов")
    p.add_argument("--filters", type=int, default=512)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--intervals", default="whitelist.test.bed.gz")
    p.add_argument("--contig-list")
    p.add_argument("--out")
    p.add_argument("--checkpoint", help="predict: .pt; по умолчанию <tag>.best.pt")
    a = p.parse_args()
    if a.mode == "predict" and a.batch == 64:
        a.batch = 256
    (train if a.mode == "train" else predict)(a)
