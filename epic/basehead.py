#!/usr/bin/env python
"""Побуквенное уточнение поверх model_26000 — быстрая проверка без перезапуска GENA.

  python epic/basehead.py eval  A [--pred D]  # valid: FULL, FULL^w×dinuc, (FULL^w×cnn) + потолки ORACLE_* (CPU, если cnn нет)
  python epic/basehead.py train A             # CNN P(старт | ±67 п.н.) на train-контигах (без valid) + таблица dinuc
  python epic/basehead.py test  A             # CNN на template.test → predictions/A_seqcnn.npy (для assemble.py --dinuc)

Метки — из bigWig обучения (epic_train/A/bw: NaN = нет разметки), последовательность — genome.fa.
Всё в направлении транскрипции ('-' = reverse complement). Предсказания GENA: predictions/A_valid_full/<contig><strand>.npz.
"""
import argparse, os
from pathlib import Path
import numpy as np, pandas as pd, pyBigWig, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

H = Path(os.environ['GENALM_HOME'])
LUT = np.full(256, 4, np.uint8)
for i, ch in enumerate(b'ACGT'): LUT[ch] = LUT[ch + 32] = i
COMP = np.array([3, 2, 1, 0, 4], np.uint8)
fa = lambda A: H / f'epic_data/{A}/genome/genome.fa'
fp = lambda A: H / f'epic_data/{A}/for_participants'


def outdir(A):
    d = H / f'epic_train/{A}/basehead'; d.mkdir(exist_ok=True); return d

def fai(A):
    return pd.read_csv(f'{fa(A)}.fai', sep='\t', header=None, index_col=0)

def valid_contigs(A):
    return (H / f'epic_train/{A}/intervals/valid_contigs.txt').read_text().split()

def train_contigs(A, max_bp, seed=0):
    vc = set(valid_contigs(A))
    cs = [c for c in pd.read_csv(fp(A) / 'template.train.bed.gz', sep='\t', header=None, usecols=[0])[0].unique() if c not in vc]
    L = fai(A)[1]; np.random.default_rng(seed).shuffle(cs)
    keep = np.cumsum([L[c] for c in cs]) <= max_bp; keep[0] = True
    return [c for c, k in zip(cs, keep) if k]

def read_contigs(A, names):
    idx, out = fai(A), {}
    with open(fa(A), 'rb') as f:
        for n in names:
            L, off, lb, lw = (int(x) for x in idx.loc[n, [1, 2, 3, 4]])
            f.seek(off); raw = np.frombuffer(f.read((L // lb + 1) * lw), np.uint8)
            out[n] = LUT[raw[raw > 32][:L]]
    return out

def oriented(A, contigs, presence=False):
    """{(contig, strand): (seq, lab)}; lab — float (NaN = нет разметки) или int8 -1/0/1 при presence."""
    seqs = read_contigs(A, contigs)
    bw = {st: pyBigWig.open(str(H / f'epic_train/{A}/bw/{A}.pool.{n}.bw')) for st, n in (('+', 'forward'), ('-', 'reverse'))}
    out = {}
    for c in contigs:
        for st in '+-':
            s = seqs[c]; y = bw[st].values(c, 0, len(s), numpy=True).astype(np.float32)
            if presence: y = np.where(np.isnan(y), -1, y > 0).astype(np.int8)
            if st == '-': s, y = COMP[s[::-1]], y[::-1].copy()
            out[c, st] = s, y
    return out


# ---------- dinuc: P(старт | предыдущий, текущий) по train-контигам ----------
def dinuc_keys(s):
    return np.r_[4, s[:-1]].astype(np.int64) * 5 + s

def get_dinuc(A, data=None):
    f = outdir(A) / 'dinuc.npy'
    if f.exists(): return np.load(f)
    data = data or oriented(A, train_contigs(A, 4e8), presence=True)
    pos, tot = np.zeros(25), np.zeros(25)
    for s, y in data.values():
        k = dinuc_keys(s)
        tot += np.bincount(k[y >= 0], minlength=25); pos += np.bincount(k[y == 1], minlength=25)
    t = (pos + 1) / (tot + 2); np.save(f, t); return t


# ---------- CNN по последовательности ----------
class SeqCNN(nn.Module):  # рецептивное поле ±67 п.н.
    def __init__(self, d=64, dil=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.inp = nn.Conv1d(5, d, 9, padding=4)
        self.blocks = nn.ModuleList(nn.Sequential(nn.BatchNorm1d(d), nn.GELU(), nn.Conv1d(d, d, 3, padding=r, dilation=r)) for r in dil)
        self.out = nn.Conv1d(d, 1, 1)

    def forward(self, x):
        h = self.inp(F.one_hot(x.long(), 5).float().transpose(1, 2))
        for b in self.blocks: h = h + b(h)
        return self.out(h)[:, 0]

def load_cnn(A, tag='cnn'):
    f = outdir(A) / f'{tag}.pt'
    if not f.exists(): return None
    m = SeqCNN().cuda(); m.load_state_dict(torch.load(f, weights_only=True)); return m.eval()


# ---------- токены GENA (для обучения «внутри токена») ----------
def token_ids(tok, s, chunk=20000):
    """Номер токена для каждого основания; токенизация кусками по 20 т.п.н. ≈ окнам predict.py."""
    L = np.frombuffer(b'ACGTN', np.uint8)
    txt = [L[s[a:a + chunk]].tobytes().decode() for a in range(0, len(s), chunk)]
    out, k = [], 0
    for t, off in zip(txt, tok(txt, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']):
        st = np.array([o[0] for o in off])
        out.append(np.maximum(np.searchsorted(st, np.arange(len(t)), 'right') - 1, 0) + k); k += len(st)
    return np.concatenate(out)

def cond_labels(tok, s, y):
    """Оставить метки только в позитивных токенах (все основания размечены, ≥1 старт), остальное -1."""
    tid = token_ids(tok, s)
    keep = ((np.bincount(tid, (y == 1).astype(float)) > 0) & (np.bincount(tid, (y < 0).astype(float)) == 0))[tid]
    return np.where(keep, y, -1).astype(np.int8)

def tokmean(x, tid):
    return (np.bincount(tid, x) / np.bincount(tid))[tid]

@torch.no_grad()
def predict_seq(model, s, C=1 << 20, pad=256):
    out = np.empty(len(s), np.float32)
    for a in range(0, len(s), C):
        lo, n = max(0, a - pad), min(C, len(s) - a)
        x = torch.from_numpy(s[lo:a + n + pad])[None].cuda()
        out[a:a + n] = torch.sigmoid(model(x))[0, a - lo:a - lo + n].cpu().numpy()
    return out

def batches(data, L, B, rng):
    keys = [k for k, (s, _) in data.items() if len(s) > L]
    w = np.array([len(data[k][0]) for k in keys], float); w /= w.sum()
    while True:
        X, Y = np.empty((B, L), np.uint8), np.empty((B, L), np.int8)
        for i, j in enumerate(rng.choice(len(keys), B, p=w)):
            s, y = data[keys[j]]; a = rng.integers(len(s) - L)
            X[i], Y[i] = s[a:a + L], y[a:a + L]
        yield torch.from_numpy(X).cuda(), torch.from_numpy(Y).cuda()

def cmd_train(a):
    data = oriented(a.A, train_contigs(a.A, a.max_bp), presence=True)
    get_dinuc(a.A, data)
    if a.cond:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer or H / 'models/moderngena_large_8192')
        data = {k: (s, cond_labels(tok, s, y)) for k, (s, y) in data.items()}
        n = sum((y >= 0).sum() for _, y in data.values()); p = sum((y == 1).sum() for _, y in data.values())
        print(f'внутри позитивных токенов: {n:,} оснований, стартов {p / n:.1%}', flush=True)
    model = SeqCNN().cuda()
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps)
    for step, (x, y) in zip(range(a.steps), batches(data, a.len, a.bs, np.random.default_rng(0))):
        m = y >= 0
        if not m.any(): continue
        loss = F.binary_cross_entropy_with_logits(model(x)[m], (y[m] == 1).float())
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if step % 500 == 0: print(step, round(loss.item(), 5), flush=True)
    torch.save(model.state_dict(), outdir(a.A) / f'{a.tag}.pt')


# ---------- оценка на valid ----------
def load_pred(d, c, st, n, key='track'):
    """(оценка, токенный трек) в направлении транскрипции; key — ключ npz или произведение ключей 'track*m'.
    Токенный трек (для границ токенов) — ключ 'tok' (predict_base.py), иначе 'track' (predict.py)."""
    z = np.load(Path(d) / f'{c}{st}.npz'); lo = int(z['lo'])
    def full(t):
        v = np.full(n, np.nan, np.float32); v[lo:lo + len(t)] = t
        return v if st == '+' else v[::-1]
    return full(np.prod([z[k] for k in key.split('*')], 0)), full(z['tok'] if 'tok' in z.files else z['track'])

def sub(x):  # как в сабмите: [0, 1], 5 знаков
    return np.round(x / x.max(), 5)

def metrics(s, y):
    pos = y > 0
    return average_precision_score(pos, s), spearmanr(s[pos], y[pos])[0]

def cmd_eval(a):
    pred = a.pred or H / f'predictions/{a.A}_valid_full'
    data, dt, cnn = oriented(a.A, valid_contigs(a.A)), get_dinuc(a.A), load_cnn(a.A, a.tag)
    col = {k: [] for k in ('y', 'full', 'dn', 'tok', 'alloc', 'dnN', 'cnn', 'cnnN')}
    for (c, st), (s, lab) in data.items():
        p, pt = load_pred(pred, c, st, len(s), a.key)
        tid = np.cumsum(np.r_[True, pt[1:] != pt[:-1]]) - 1        # токены модели = серии одинаковых значений
        y0 = np.nan_to_num(lab)
        ts, tl = np.bincount(tid, weights=y0)[tid], np.bincount(tid)[tid]
        v = dict(y=y0, full=p, dn=dt[dinuc_keys(s)], tok=ts,
                 alloc=pt * np.where(ts > 0, y0 / np.maximum(ts, 1), 1 / tl))  # токены FULL + идеальное распределение внутри токена
        v['dnN'] = v['dn'] / tokmean(v['dn'], tid)                       # только перераспределение внутри токена
        if cnn is not None: v['cnn'] = predict_seq(cnn, s); v['cnnN'] = v['cnn'] / tokmean(v['cnn'], tid)
        m = ~np.isnan(lab) & ~np.isnan(p)
        for k, x in v.items(): col[k].append(x[m].astype(np.float32))
    C = {k: np.concatenate(v) for k, v in col.items() if v}
    y, f, dn = C['y'], C['full'], C['dn']
    S = {'dinuc': dn, 'FULL': f, **{f'FULL^{w}×dinuc': f ** w * dn for w in a.w}}
    if 'cnn' in C: S |= {'cnn': C['cnn'], **{f'FULL^{w}×cnn': f ** w * C['cnn'] for w in a.w}}
    for x in ('dnN', 'cnnN'):
        if x in C: S |= {f'FULL^{w}×{x}^{e}': f ** w * C[x] ** e for w in a.w for e in a.a}
    print(f'{"":22s} AUPRC   Spearman   позиций {len(y):,}, позитивов {np.mean(y > 0):.3%}')
    for k, s in S.items(): print(f'{k:22s} %.4f  %.4f' % metrics(sub(s), y), flush=True)
    for k, s in {'ORACLE_tok': C['tok'], 'ORACLE_tok×dinuc': C['tok'] * dn, 'FULL×ORACLE_alloc': C['alloc']}.items():
        print(f'{k:22s} %.4f  %.4f' % metrics(s, y), flush=True)   # потолки, без округления


# ---------- test → npy в порядке template.test ----------
def cmd_test(a):
    bed = pd.read_csv(fp(a.A) / 'template.test.bed.gz', sep='\t', header=None, usecols=[0, 1, 2, 5])
    bed.columns = ['c', 's', 'e', 'st']
    n = (bed.e - bed.s).to_numpy(); bed['o'] = np.cumsum(n) - n
    out, cnn, seqs = np.zeros(n.sum(), np.float32), load_cnn(a.A, a.tag), read_contigs(a.A, bed.c.unique())
    for (c, st), g in bed.groupby(['c', 'st'], sort=False):
        print(c, st, flush=True)
        v = predict_seq(cnn, seqs[c] if st == '+' else COMP[seqs[c][::-1]])
        v = v if st == '+' else v[::-1]
        for b0, e0, o in zip(g.s, g.e, g.o): out[o:o + e0 - b0] = v[b0:e0]
    f = H / f'predictions/{a.A}_{a.tag}.npy'; np.save(f, out); print(f, len(out))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['train', 'eval', 'test']); ap.add_argument('A')
    ap.add_argument('--pred'); ap.add_argument('--w', type=float, nargs='+', default=[1, 1.5, 2, 3])
    ap.add_argument('--steps', type=int, default=8000); ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--len', type=int, default=4096); ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--max-bp', type=float, default=4e8)
    ap.add_argument('--tag', default='cnn'); ap.add_argument('--cond', action='store_true'); ap.add_argument('--tokenizer')
    ap.add_argument('--a', type=float, nargs='+', default=[1, 2])
    ap.add_argument('--key', default='track')   # ключ npz (или произведение ключей через '*')
    a = ap.parse_args(); globals()['cmd_' + a.cmd](a)
