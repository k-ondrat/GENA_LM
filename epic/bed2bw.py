"""csRNA EPIC -> стрэндовый bigWig из меток train (ровно то, что скорят).

    python epic/bed2bw.py ASSEMBLY [--data epic_data] [--out epic_train]

Вход:  for_participants/template.train.bed.gz (contig, start, end, ., ., strand; сначала все +, потом все -)
       for_participants/csRNA-train.txt.gz   (одно число на позицию шаблона = r1 + r2)
       genome/genome.fa(.gz)                 (размеры контигов, как их видит датасет)
Выход: epic_train/A/bw/A.pool.{forward,reverse}.bw
  * интервалы покрывают ТОЛЬКО позиции шаблона, нули записаны явно;
  * вне шаблона интервалов нет (= нет разметки, см. csrna_dataset.py);
  * подряд идущие равные значения сжаты в один интервал (сумма по токену не меняется).
"""
import argparse, os
import numpy as np, pandas as pd, pyBigWig, pysam

p = argparse.ArgumentParser()
p.add_argument("assembly")
p.add_argument("--data", default=os.path.join(os.environ["GENALM_HOME"], "epic_data"))
p.add_argument("--out", default=os.path.join(os.environ["GENALM_HOME"], "epic_train"))
a = p.parse_args()
A = a.assembly
fp = os.path.join(a.data, A, "for_participants")
gdir = os.path.join(a.data, A, "genome")

fa = next(f for f in (f"{gdir}/genome.fa", f"{gdir}/genome.fa.gz") if os.path.exists(f))
with pysam.FastaFile(fa) as F:
    sizes = list(zip(F.references, F.lengths))
csize = dict(sizes)

t = pd.read_csv(f"{fp}/template.train.bed.gz", sep="\t", header=None, usecols=[0, 1, 2, 5],
                names=["c", "s", "e", "st"])
t["L"] = t.e - t.s
t["off"] = np.r_[0, np.cumsum(t.L.to_numpy())[:-1]]
vals = pd.read_csv(f"{fp}/csRNA-train.txt.gz", header=None, dtype=np.int32).iloc[:, 0].to_numpy()
assert len(vals) == t.L.sum(), (len(vals), t.L.sum())
assert set(t.c) <= set(csize), f"контиги шаблона нет в fasta: {set(t.c) - set(csize)}"

def runs(g):
    """Интервалы шаблона одного контига и цепи -> RLE-записи (start, end, value)."""
    g = g.sort_values("s")
    s, e, L, off = (g[k].to_numpy() for k in ("s", "e", "L", "off"))
    assert (s[1:] >= e[:-1]).all(), "интервалы шаблона перекрываются"
    pos = np.repeat(s - np.r_[0, np.cumsum(L)[:-1]], L) + np.arange(L.sum())
    v = np.concatenate([vals[o:o + l] for o, l in zip(off, L)])
    brk = np.flatnonzero((v[1:] != v[:-1]) | (pos[1:] != pos[:-1] + 1)) + 1
    i0 = np.r_[0, brk]
    i1 = np.r_[brk, len(v)]
    return pos[i0], pos[i1 - 1] + 1, v[i0].astype(float)

os.makedirs(f"{a.out}/{A}/bw", exist_ok=True)
for st, name in (("+", "forward"), ("-", "reverse")):
    path = f"{a.out}/{A}/bw/{A}.pool.{name}.bw"
    bw = pyBigWig.open(path, "w")
    bw.addHeader(sizes)
    ts = t[t.st == st]
    total = 0.0
    for c, _ in sizes:                          # записи - в порядке заголовка
        g = ts[ts.c == c]
        if len(g) == 0:
            continue
        s0, e0, v0 = runs(g)
        bw.addEntries([c] * len(s0), s0.tolist(), ends=e0.tolist(), values=v0.tolist())
        total += float((v0 * (e0 - s0)).sum())
    bw.close()
    exp = int(sum(vals[o:o + l].sum() for o, l in zip(ts.off, ts.L)))
    print(f"{name}: сумма в bw {total:.0f}, в метках {exp} -> {path}")

# проверка: нули читаются как интервалы, вне шаблона - nan, значения совпадают с метками
bw = pyBigWig.open(f"{a.out}/{A}/bw/{A}.pool.forward.bw")
r = t[t.st == "+"].iloc[0]
w = min(int(r.L), 2000)
got = np.nan_to_num(bw.values(r.c, int(r.s), int(r.s) + w))
print("значения первого интервала совпадают:", np.array_equal(got, vals[r.off:r.off + w]),
      "| нулевых интервалов в нём:", sum(x[2] == 0 for x in bw.intervals(r.c, int(r.s), int(r.s) + w)))
if r.s > 0:
    print("позиция перед интервалом (ожидаем nan):", bw.values(r.c, int(r.s) - 1, int(r.s)))
bw.close()
