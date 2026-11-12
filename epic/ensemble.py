#!/usr/bin/env python
"""Шаг 2: ансамбль GENA-головы и референсной CNN на valid-контигах, метрика лидерборда.

  python epic/ensemble.py A --gena DIR --cnn DIR [--w 0 0.25 0.5 0.75 1] [--alphas 0.6 0.7 0.8]

z = (1−w)·log λ_GENA + w·log λ_CNN;  λ_GENA = expm1(track) (голова учит log1p),  λ_CNN = exp(lograte).
w = 0 — только GENA, w = 1 — только CNN. Шкала сабмита rank^(-α), ранг × N_test/N_valid (как lbq.py).
"""
import argparse
import numpy as np, pyBigWig
from basehead import fai, valid_contigs, H
from lbq import lb, ranks, quant, n_test


def load(A, gd, cd):
    L = fai(A)[1]
    bw = {st: pyBigWig.open(f'{H}/epic_train/{A}/bw/{A}.pool.{n}.bw') for st, n in (('+', 'forward'), ('-', 'reverse'))}
    Y, G, C = [], [], []
    for c in valid_contigs(A):
        for st in '+-':
            y = bw[st].values(c, 0, int(L[c]), numpy=True).astype(np.float32)
            t = []
            for d, key in ((gd, 'track'), (cd, 'lograte')):
                z = np.load(f'{d}/{c}{st}.npz'); s = np.full(len(y), np.nan, np.float32)
                v, lo = z[key], int(z['lo']); s[lo:lo + len(v)] = v; t.append(s)
            m = np.isfinite(y) & np.isfinite(t[0]) & np.isfinite(t[1])
            Y.append(y[m]); G.append(t[0][m]); C.append(t[1][m])
    g = np.log(np.expm1(np.clip(np.concatenate(G), 0, None)) + 1e-12).astype(np.float32)
    return np.concatenate(Y), g, np.concatenate(C)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('A'); ap.add_argument('--gena', required=True); ap.add_argument('--cnn', required=True)
    ap.add_argument('--w', type=float, nargs='+', default=[0, 0.25, 0.4, 0.5, 0.6, 0.75, 1])
    ap.add_argument('--alphas', type=float, nargs='+', default=[0.6, 0.65, 0.7, 0.8])
    a = ap.parse_args()
    y, g, c = load(a.A, a.gena, a.cnn)
    N = n_test(a.A)
    print(f'{a.A}: позиций {len(y):,}, позитивов {(y > 0).mean():.4%}; corr(log λ GENA, log λ CNN) = '
          f'{np.corrcoef(g[::97], c[::97])[0, 1]:.3f}', flush=True)
    print('   w   ' + '   '.join(f'α {al:<13g}' for al in a.alphas))
    for w in a.w:
        r = ranks((1 - w) * g + w * c, N)
        print(f'{w:5.2f}  ' + '   '.join('%.4f / %.4f' % lb(quant(r, al), y) for al in a.alphas), flush=True)
