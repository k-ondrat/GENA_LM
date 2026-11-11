#!/usr/bin/env python
"""Метрики лидерборда EPIC и квантование сабмита rank^(-alpha).

  python epic/lbq.py A --pred DIR [--alphas 0.3 0.4 ...]   # перебор шкал на valid-контигах

lb()    — ровно как EPIC_scoring/scorer.py (--streaming): AUPRC по гистограмме из rint(v*1e5),
          Spearman = pearsonr(dense-ранги v, dense-ранги y) только по позитивам.
quant() — уровень = rank^(-alpha), 5 знаков. На valid ранг масштабируется на N_test/N_valid,
          чтобы число различимых уровней в верхушке было как на test.
"""
import argparse, json, os
import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

H = os.environ['GENALM_HOME']
SCALE = 100_000


def lb(v, y):
    """(AUPRC, Spearman) лидерборда; v — сабмит в [0, 1], y — счётчики."""
    b, p = np.rint(v * SCALE).astype(np.int64), y > 0
    P = np.bincount(b[p], minlength=SCALE + 1)[::-1]
    N = np.bincount(b[~p], minlength=SCALE + 1)[::-1]
    tp, seen, k = np.cumsum(P), np.cumsum(P + N), P > 0
    ap = float(np.sum(tp[k] / seen[k] * P[k]) / P.sum())
    return ap, float(pearsonr(rankdata(b[p], 'dense'), rankdata(y[p], 'dense'))[0])


def ranks(s, n_ref=None):
    """Ранг по убыванию (ничьи — общий ранг); n_ref — длина, под которую масштабировать."""
    r = rankdata(-np.nan_to_num(s, nan=-np.inf), 'min')
    return r * (n_ref / len(s)) if n_ref else r


def quant(r, alpha):
    return np.round(np.maximum(r, 1) ** -alpha, 5)


def n_test(A):
    return json.load(open(f'{H}/epic_data/genome_metrics.json'))['assemblies'][A]['template']['test']['length']


def load_valid(A, pred, key='track'):
    """Выход модели и счётчики (bigWig обучения) на valid-контигах, только размеченные позиции."""
    import pyBigWig
    from basehead import fai, valid_contigs
    L = fai(A)[1]
    bw = {st: pyBigWig.open(f'{H}/epic_train/{A}/bw/{A}.pool.{n}.bw') for st, n in (('+', 'forward'), ('-', 'reverse'))}
    S, Y = [], []
    for c in valid_contigs(A):
        for st in '+-':
            y = bw[st].values(c, 0, int(L[c]), numpy=True).astype(np.float32)
            s = np.full(len(y), np.nan, np.float32)
            z = np.load(f'{pred}/{c}{st}.npz'); t, lo = z[key], int(z['lo']); s[lo:lo + len(t)] = t
            m = np.isfinite(y) & np.isfinite(s)
            S.append(s[m]); Y.append(y[m])
    return np.concatenate(S), np.concatenate(Y)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('A'); ap.add_argument('--pred', required=True); ap.add_argument('--key', default='track')
    ap.add_argument('--alphas', type=float, nargs='+', default=[0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument('--tol', type=float, default=0.995, help='допустимая доля лучшего AUPRC')
    a = ap.parse_args()

    s, y = load_valid(a.A, a.pred, a.key)
    s = np.clip(s, 0, None); p = y > 0
    print(f'{a.A}: позиций {len(y):,}, позитивов {p.sum():,} ({p.mean():.4%}), N_test {n_test(a.A):,}', flush=True)
    rows = {'cur log1p': np.round(np.log1p(s) / np.log1p(s).max(), 5),       # как assemble.py сейчас
            'lin expm1': np.round(np.expm1(s) / np.expm1(s).max(), 5)}
    r = ranks(s, n_test(a.A))
    rows |= {f'rank^-{x:g}': quant(r, x) for x in a.alphas}
    res = {k: lb(v, y) for k, v in rows.items()}
    print(f'{"":12s} AUPRC   Spearman_LB  (scipy)')
    for k, (au, sp) in res.items():
        print(f'{k:12s} {au:.4f}  {sp:.4f}       ({spearmanr(rows[k][p], y[p])[0]:.4f})', flush=True)
    top = max(au for au, _ in res.values())
    ok = {k: v for k, v in res.items() if k.startswith('rank') and v[0] >= a.tol * top}
    best = max(ok, key=lambda k: ok[k][1])
    print(f'ALPHA {a.A} {best.split("^-")[1]}   # AUPRC {ok[best][0]:.4f}, Spearman_LB {ok[best][1]:.4f}')
