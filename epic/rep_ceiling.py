#!/usr/bin/env python
"""Шаг 2: потолок по репликам — насколько одна реплика csRNA предсказывает другую (раздел 3 стратегии).

  python epic/rep_ceiling.py A [--alphas 0.6 0.7 0.8]

Реплики: for_participants/csRNA-*-r1-train.bed.gz и -r2- (позиция = интервал bed, счётчик в 5-м столбце;
интервал длиной > 1 — серия позиций с одним счётчиком). Вселенная — все позиции template.train (обе цепи),
нули вне bed. Метрики как у скорера (lbq.lb): AUPRC по «правде > 0» с учётом ничьих; Spearman — Пирсон
dense-рангов по позитивам правды. «сырые» — счётчик реплики как есть; «α» — шкала сабмита rank^(-α),
ранг × N_test/N_train (как lbq.py).
"""
import argparse, glob, gzip, json, os
import numpy as np, pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

H = os.environ['GENALM_HOME']


def tlen(A, sub):
    return json.load(open(f'{H}/epic_data/genome_metrics.json'))['assemblies'][A]['template'][sub]['length']


def read_bed(path, cid):
    d = pd.read_csv(path, sep='\t', header=None, usecols=[0, 1, 2, 4, 5], names=['c', 's', 'e', 'n', 'st'])
    L = (d.e - d.s).to_numpy()
    pos = np.repeat(d.s.to_numpy(), L) + (np.arange(L.sum()) - np.repeat(np.cumsum(L) - L, L))
    base = (d.c.map(cid).to_numpy() * 2 + (d.st == '-').to_numpy()).astype(np.int64)
    return np.repeat(base, L) << 34 | pos, np.repeat(d.n.to_numpy(), L).astype(np.float64)


def load_reps(A):
    """Объединение позиций r1 ∪ r2: ключ (контиг·2 + цепь) << 34 | позиция, счётчики r1, r2."""
    fp = f'{H}/epic_data/{A}/for_participants'
    f1, f2 = (sorted(glob.glob(f'{fp}/csRNA-*-r{i}-train.bed.gz'))[0] for i in (1, 2))
    names = pd.unique(pd.concat([pd.read_csv(f, sep='\t', header=None, usecols=[0])[0] for f in (f1, f2)]))
    cid = {c: i for i, c in enumerate(names)}
    (k1, n1), (k2, n2) = read_bed(f1, cid), read_bed(f2, cid)
    key = np.union1d(k1, k2)
    r1, r2 = np.zeros(len(key)), np.zeros(len(key))
    r1[np.searchsorted(key, k1)] += n1; r2[np.searchsorted(key, k2)] += n2
    return key, r1, r2, names, (os.path.basename(f1), os.path.basename(f2))


def ap(level, truth, n_imp, lvl0):
    """AUPRC с ничьими: явные позиции (level, truth) + n_imp негативов на уровне lvl0."""
    u, inv = np.unique(np.r_[level, lvl0], return_inverse=True)
    P = np.bincount(inv, weights=np.r_[truth, 0], minlength=len(u))
    N = np.bincount(inv, weights=np.r_[~truth, 0], minlength=len(u)); N[inv[-1]] += n_imp
    P, N = P[::-1], N[::-1]
    tp, seen, k = np.cumsum(P), np.cumsum(P + N), P > 0
    return float(np.sum(tp[k] / seen[k] * P[k]) / P.sum())


def metrics(pred, truth, n_tot, n_test, alpha=None):
    t = truth > 0
    if alpha is None: lv, lvl0 = pred, 0.0
    else:
        nz = pred > 0
        r = np.full(len(pred), nz.sum() + 1.0)                     # нули — общий ранг ниже всех ненулевых
        r[nz] = rankdata(-pred[nz], 'min')
        q = lambda x: np.round(np.maximum(x * n_test / n_tot, 1) ** -alpha, 5)
        lv, lvl0 = q(r), float(q(np.float64(nz.sum() + 1)))
    sp = pearsonr(rankdata(lv[t], 'dense'), rankdata(truth[t], 'dense'))[0]
    return ap(lv, t, n_tot - len(pred), lvl0), sp


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('A'); p.add_argument('--alphas', type=float, nargs='+', default=[0.6, 0.7, 0.8])
    a = p.parse_args()
    key, r1, r2, _, files = load_reps(a.A)
    n_tot, n_test = tlen(a.A, 'train'), tlen(a.A, 'test')
    print(f'{a.A}: {files[0]}, {files[1]} | позиций шаблона {n_tot:,} | позитивов r1 {(r1 > 0).sum():,}, r2 {(r2 > 0).sum():,}, '
          f'r1+r2 {len(key):,} ({len(key) / n_tot:.4%}) | ридов r1 {r1.sum():,.0f}, r2 {r2.sum():,.0f}')
    print('предсказание → правда    ' + '   '.join(['сырые            '] + [f'α {x:<14g}' for x in a.alphas]) + '   scipy-ρ')
    for nm, x, y in (('r1 → r2', r1, r2), ('r2 → r1', r2, r1)):
        cols = [metrics(x, y, n_tot, n_test)] + [metrics(x, y, n_tot, n_test, al) for al in a.alphas]
        print(f'{nm:24s} ' + '   '.join('%.4f / %.4f' % c for c in cols) + f'   {spearmanr(x[y > 0], y[y > 0])[0]:.4f}')
