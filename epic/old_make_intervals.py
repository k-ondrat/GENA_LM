"""Окна для дообучения: сплит контигов train/valid + центры окон по обеим цепям.

    python epic/make_intervals.py ASSEMBLY [--valid-frac 0.1 --step 25000 --top 20000 ...]

Окно = TSSWindow вокруг центра: num_before токенов выше и num_after ниже (~±24 т.п.н.).
Центры:
  * регулярные: шаг --step по каждому контигу и цепи; окно берётся, если размечено
    (покрыто template.train этой цепи) >= --min-labeled его длины;
  * сильные TSS: топ позиций по csRNA (одна на бин 1 т.п.н.), центр сдвинут на
    случайные ±--jitter, чтобы старт оказывался в разных местах окна.
Сплит по контигам: случайные контиги, пока не наберётся --valid-frac размеченной длины.
Выход: epic_train/A/intervals/A.{train,valid}.{forward,reverse}.tsv + valid_contigs.txt
       колонки chromosome, TSS (центр), TES (не используется TSSWindow), gene_id, kind
"""
import argparse, os
import numpy as np, pandas as pd, pysam

p = argparse.ArgumentParser()
p.add_argument("assembly")
p.add_argument("--data", default=os.path.join(os.environ["GENALM_HOME"], "epic_data"))
p.add_argument("--out", default=os.path.join(os.environ["GENALM_HOME"], "epic_train"))
p.add_argument("--valid-frac", type=float, default=0.1)
p.add_argument("--step", type=int, default=25000)
p.add_argument("--half", type=int, default=24000, help="полуширина окна в п.н. для доли разметки")
p.add_argument("--min-labeled", type=float, default=0.2)
p.add_argument("--tss-ratio", type=float, default=1.0,
               help="TSS-окон на одно регулярное, отдельно для каждого сплита и цепи")
p.add_argument("--min-count", type=int, default=3, help="минимум ридов (r1+r2) у TSS-центра")
p.add_argument("--max-train", type=int, default=25000, help="потолок окон train на цепь и тип")
p.add_argument("--max-valid", type=int, default=250, help="потолок окон valid на цепь и тип (кривая loss)")
p.add_argument("--jitter", type=int, default=10000)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
A, rng = a.assembly, np.random.default_rng(a.seed)
fp = os.path.join(a.data, A, "for_participants")
gdir = os.path.join(a.data, A, "genome")

fa = next(f for f in (f"{gdir}/genome.fa", f"{gdir}/genome.fa.gz") if os.path.exists(f))
with pysam.FastaFile(fa) as F:
    csize = dict(zip(F.references, F.lengths))

t = pd.read_csv(f"{fp}/template.train.bed.gz", sep="\t", header=None, usecols=[0, 1, 2, 5],
                names=["c", "s", "e", "st"])
t["L"] = t.e - t.s
off = np.r_[0, np.cumsum(t.L.to_numpy())[:-1]]
vals = pd.read_csv(f"{fp}/csRNA-train.txt.gz", header=None, dtype=np.int32).iloc[:, 0].to_numpy()
assert len(vals) == t.L.sum()

# --- сплит контигов
lab = t.groupby("c").L.sum()
order = rng.permutation(lab.index.to_numpy())
cum = lab[order].cumsum().to_numpy()
valid = set(order[: np.searchsorted(cum, a.valid_frac * lab.sum()) + 1])
assert len(valid) < len(lab), "все контиги ушли в valid"

# --- доля разметки в окне: префиксные суммы по интервалам (контиг, цепь)
cov = {}
for (c, st), g in t.groupby(["c", "st"]):
    g = g.sort_values("s")
    s, e = g.s.to_numpy(), g.e.to_numpy()
    cov[c, st] = (s, e, np.r_[0, np.cumsum(e - s)])

def labeled(c, st, lo, hi):
    s, e, cl = cov[c, st]
    def F(x):
        k = np.searchsorted(s, x, "right") - 1
        kk = np.maximum(k, 0)
        return np.where(k >= 0, cl[kk] + np.clip(x - s[kk], 0, e[kk] - s[kk]), 0)
    return F(hi) - F(lo)

split_of = lambda c: np.where(pd.Series(c).isin(valid), "valid", "train")

# --- регулярные центры
reg = []
for (c, st), (s, e, _) in cov.items():
    x = np.arange(s.min(), e.max(), a.step)
    frac = labeled(c, st, x - a.half, x + a.half) / (2 * a.half)
    reg.append(pd.DataFrame({"c": c, "st": st, "x": x[frac >= a.min_labeled], "kind": "regular"}))
reg = pd.concat(reg, ignore_index=True)
n_reg = reg.assign(split=split_of(reg.c)).groupby(["split", "st"]).size()
rows = [reg]

# --- сильные TSS: столько же (x tss-ratio), сколько регулярных, в каждом сплите и цепи
nz = np.flatnonzero(vals)
r = np.searchsorted(off, nz, "right") - 1
top = pd.DataFrame({"c": t.c.to_numpy()[r], "st": t.st.to_numpy()[r],
                    "pos": t.s.to_numpy()[r] + nz - off[r], "v": vals[nz]})
top = top[top.v >= a.min_count]
top["bin"] = top.pos // 1000
top = top.sort_values("v", ascending=False).drop_duplicates(["c", "st", "bin"])
top["split"] = split_of(top.c)
top = pd.concat(g.head(int(round(a.tss_ratio * n_reg.get((sp, st), 0))))
                for (sp, st), g in top.groupby(["split", "st"]))
print("минимальный счётчик в отобранных TSS по сплиту/цепи:",
      top.groupby(["split", "st"]).v.min().to_dict())
top["x"] = top.pos + rng.integers(-a.jitter, a.jitter + 1, len(top))
rows.append(top.assign(kind="tss")[["c", "st", "x", "kind"]])

w = pd.concat(rows, ignore_index=True)
w["x"] = np.clip(w.x, 0, w.c.map(csize) - 1).astype(np.int64)
w["gene_id"] = w.c + ":" + w.x.astype(str) + ":" + np.where(w.st == "+", "f", "r")
w = w.drop_duplicates("gene_id")
w["split"] = np.where(w.c.isin(valid), "valid", "train")
cap = {"train": a.max_train, "valid": a.max_valid}
w = pd.concat(g.sample(min(len(g), cap[sp]), random_state=a.seed)
              for (sp, _, _), g in w.groupby(["split", "st", "kind"]))

od = f"{a.out}/{A}/intervals"
os.makedirs(od, exist_ok=True)
pd.Series(sorted(valid)).to_csv(f"{od}/valid_contigs.txt", index=False, header=False)
for (sp, st), g in w.groupby(["split", "st"]):
    name = "forward" if st == "+" else "reverse"
    out = g.sort_values(["c", "x"]).rename(columns={"c": "chromosome", "x": "TSS"})
    out["TES"] = out.TSS + 1
    out[["chromosome", "TSS", "TES", "gene_id", "kind"]].to_csv(
        f"{od}/{A}.{sp}.{name}.tsv", sep="\t", index=False)

print(f"valid: {len(valid)} контигов, {lab[list(valid)].sum() / lab.sum():.1%} размеченной длины")
print(w.groupby(["split", "st", "kind"]).size().unstack("kind").to_string())
