"""AUPRC / Spearman на valid-контигах по меткам csRNA-train.txt (как в скорере EPIC).

    python epic/eval_valid.py ASSEMBLY --pred A=DIR_A B=DIR_B ATAC=DIR_ATAC [--w 0.5 1 2]

DIR - каталоги predict.py (<contig><strand>.npz). Для каждого трека считаются: сам трек
и трек^w x dinuc. Динуклеотид пересчитывается ТОЛЬКО по train-контигам (valid исключены).
Все варианты оцениваются на одних и тех же позициях (где посчитаны все треки);
оценки нормируются в [0, 1] и округляются до 5 знаков, как в сабмите (ничьи как у скорера).
"""
import argparse, os
import numpy as np, pandas as pd, pysam
from scipy.stats import spearmanr

H = os.environ["GENALM_HOME"]
p = argparse.ArgumentParser()
p.add_argument("assembly")
p.add_argument("--pred", nargs="+", default=[], help="ИМЯ=КАТАЛОГ")
p.add_argument("--w", nargs="+", type=float, default=[0.5, 1.0, 2.0])
p.add_argument("--contig-list")
a = p.parse_args()
A = a.assembly
fp = f"{H}/epic_data/{A}/for_participants"
valid = set(open(a.contig_list or f"{H}/epic_train/{A}/intervals/valid_contigs.txt").read().split())
preds = dict(x.split("=", 1) for x in a.pred)

t = pd.read_csv(f"{fp}/template.train.bed.gz", sep="\t", header=None, usecols=[0, 1, 2, 5],
                names=["c", "s", "e", "st"])
t["L"] = t.e - t.s
t["off"] = np.r_[0, np.cumsum(t.L.to_numpy())[:-1]]
vals = pd.read_csv(f"{fp}/csRNA-train.txt.gz", header=None, dtype=np.int32).iloc[:, 0].to_numpy()
fa = pysam.FastaFile(f"{H}/epic_data/{A}/genome/genome.fa")
LUT = np.full(256, 4, np.uint8)
for i, ch in enumerate(b"ACGT"):
    LUT[ch] = i
    LUT[ord(chr(ch).lower())] = i
COMP = np.array([3, 2, 1, 0, 4], np.uint8)

def positions(g):
    s, L = g.s.to_numpy(), g.L.to_numpy()
    pos = np.repeat(s - np.r_[0, np.cumsum(L)[:-1]], L) + np.arange(L.sum())
    y = np.concatenate([vals[o:o + l] for o, l in zip(g.off, L)])
    return pos, y

def dinuc_code(seq, pos, st):
    """(предыдущий, текущий) нуклеотид по направлению транскрипции -> 0..24."""
    if st == "+":
        prev, cur = seq[np.maximum(pos - 1, 0)], seq[pos]
    else:
        prev, cur = COMP[seq[np.minimum(pos + 1, len(seq) - 1)]], COMP[seq[pos]]
    return prev.astype(np.int64) * 5 + cur

# --- динуклеотид по train-контигам
tot, hit = np.zeros(25), np.zeros(25)
for c, gc in t[~t.c.isin(valid)].groupby("c"):
    seq = LUT[np.frombuffer(fa.fetch(c).encode(), np.uint8)]
    for st, g in gc.groupby("st"):
        pos, y = positions(g)
        d = dinuc_code(seq, pos, st)
        tot += np.bincount(d, minlength=25)
        hit += np.bincount(d, weights=(y > 0), minlength=25)
rate = (hit + 1) / (tot + 2)

# --- valid-позиции: метки, dinuc, треки
Y, S = [], {"dinuc": []} | {k: [] for k in preds}
for c, gc in t[t.c.isin(valid)].groupby("c"):
    seq = LUT[np.frombuffer(fa.fetch(c).encode(), np.uint8)]
    for st, g in gc.groupby("st"):
        pos, y = positions(g)
        Y.append(y)
        S["dinuc"].append(rate[dinuc_code(seq, pos, st)])
        for k, d in preds.items():
            f, x = f"{d}/{c}{st}.npz", np.full(len(pos), np.nan, np.float32)
            if os.path.exists(f):
                z = np.load(f); tr, lo = z["track"], int(z["lo"])
                ok = (pos >= lo) & (pos < lo + len(tr))
                x[ok] = tr[pos[ok] - lo]
            S[k].append(x)
y = np.concatenate(Y)
S = {k: np.concatenate(v) for k, v in S.items()}
common = np.all([np.isfinite(v) for v in S.values()], axis=0)
print(f"valid-позиций: {len(y):,} | общих для всех треков: {common.mean():.1%} | позитивов: {(y[common] > 0).mean():.5f}")
for k in preds:
    print(f"  покрытие {k}: {np.isfinite(S[k]).mean():.1%}")
y = y[common]
S = {k: v[common] for k, v in S.items()}

def norm(v):
    v = np.log1p(np.clip(v, 0, None)) if v is not S["dinuc"] else v
    return v / (v.max() or 1)

def ap(pos, s):
    o = np.argsort(-s, kind="stable"); s, pos = s[o], pos[o]
    tp = np.cumsum(pos); fp = np.cumsum(~pos)
    i = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]          # последний индекс каждой группы ничьих
    tp, fp = tp[i], fp[i]
    return float(np.sum(np.diff(np.r_[0, tp / tp[-1]]) * tp / (tp + fp)))

def score(name, s):
    s = np.round(s / (s.max() or 1), 5)
    pos = y > 0
    rows.append((name, ap(pos, s), spearmanr(y[pos], s[pos])[0]))

rows, d = [], norm(S["dinuc"])
score("dinuc", d)
for k in preds:
    x = norm(S[k])
    score(k, x)
    for w in a.w:
        score(f"{k}^{w:g} x dinuc", x ** w * d)
print(pd.DataFrame(rows, columns=["вариант", "AUPRC", "Spearman"]).sort_values("AUPRC", ascending=False)
      .to_string(index=False, float_format="%.4f"))
