#!/usr/bin/env python
"""Шаг 2: замер картируемости (раздел 4E стратегии) — сколько позиций шаблона train не могут быть позитивом.

  python epic/mappability.py A [--k 20 58] [--threads 16]

В метки вошли только уникально картированные риды 20–58 нт. GenMap ($GENMAP, по умолчанию
$GENALM_HOME/tools/genmap/bin/genmap) считает частоту каждого k-мера по всему геному с обратной цепью
→ $GENALM_HOME/mapp/<A>/k<K>.bedgraph (значение в i — частота k-мера, начинающегося в i).
Рид длины L с 5'-концом в p: на «+» k-мер начинается в p, на «−» — в p − L + 1.
Классы позиций: «мёртвые» — неуникален даже 58-мер (уникальный рид отсюда невозможен при любой длине);
«частично» — неуникален 20-мер, но уникален 58-мер (только длинные риды). Для каждого класса —
доля позиций шаблона, позитивов (r1 + r2) и ридов.
"""
import argparse, os, subprocess
import numpy as np, pandas as pd
from rep_ceiling import load_reps, tlen

H = os.environ['GENALM_HOME']
GM = os.environ.get('GENMAP', f'{H}/tools/genmap/bin/genmap')


def run_genmap(A, ks, T):
    M = f'{H}/mapp/{A}'; os.makedirs(M, exist_ok=True)
    if not os.path.exists(f'{M}/idx'):
        subprocess.run([GM, 'index', '-F', f'{H}/epic_data/{A}/genome/genome.fa', '-I', f'{M}/idx'], check=True)
    for k in ks:
        if not os.path.exists(f'{M}/k{k}.bedgraph'):
            subprocess.run([GM, 'map', '-K', str(k), '-E', '0', '-I', f'{M}/idx', '-O', f'{M}/k{k}', '-fs', '-bg',
                            '-T', str(T)], check=True)
    return M


def nonunique(path):
    """{контиг: (начала, концы, накопленная длина)} интервалов с частотой > 1."""
    d = pd.read_csv(path, sep='\t', header=None, names=['c', 's', 'e', 'v'])
    d = d[d.v > 1]
    return {c: (g.s.to_numpy(), g.e.to_numpy(), np.r_[0, np.cumsum(g.e - g.s)]) for c, g in d.groupby('c', sort=False)}


def covered(nu, c, x):
    """Сколько позиций < x (вектор) лежит в неуникальных интервалах контига c."""
    if c not in nu: return np.zeros(len(x))
    s, e, cum = nu[c]
    j = np.searchsorted(s, x, 'right') - 1
    jj = np.clip(j, 0, None)
    return np.where(j < 0, 0, cum[jj] + np.clip(x - s[jj], 0, e[jj] - s[jj]))


def is_nu(nu, c, p):
    if c not in nu: return np.zeros(len(p), bool)
    s, e, _ = nu[c]
    j = np.searchsorted(s, p, 'right') - 1
    return (j >= 0) & (p < e[np.clip(j, 0, None)])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('A')
    ap.add_argument('--k', type=int, nargs=2, default=[20, 58]); ap.add_argument('--threads', type=int, default=16)
    a = ap.parse_args()
    M = run_genmap(a.A, a.k, a.threads)
    NU = {k: nonunique(f'{M}/k{k}.bedgraph') for k in a.k}
    t = pd.read_csv(f'{H}/epic_data/{a.A}/for_participants/template.train.bed.gz', sep='\t', header=None,
                    usecols=[0, 1, 2, 5], names=['c', 's', 'e', 'st'])
    n_tot = int((t.e - t.s).sum()); assert n_tot == tlen(a.A, 'train'), (n_tot, tlen(a.A, 'train'))
    key, r1, r2, names, _ = load_reps(a.A)
    cnt = r1 + r2; cc = names[(key >> 34) // 2]; minus = ((key >> 34) % 2).astype(bool); pos = key & (2**34 - 1)
    res = {}
    for k in a.k:
        sh = k - 1                       # на «−» k-мер рида начинается на k − 1 левее его 5'-конца
        npos = sum(covered(NU[k], c, np.r_[g.e.to_numpy(), g.s.to_numpy()] - (sh if st == '-' else 0)) @ np.r_[np.ones(len(g)), -np.ones(len(g))]
                   for (c, st), g in t.groupby(['c', 'st'], sort=False))
        nu = np.zeros(len(key), bool)
        for c in np.unique(cc):
            m = cc == c
            nu[m] = is_nu(NU[k], c, pos[m] - np.where(minus[m], sh, 0))
        res[k] = (npos, nu)
    (k1, (n1, u1)), (k2, (n2, u2)) = sorted(res.items())
    print(f'{a.A}: позиций шаблона {n_tot:,}, позитивов {len(key):,} ({len(key) / n_tot:.4%}), ридов {cnt.sum():,.0f}')
    print(f'{"класс":34s} {"позиций":>9s} {"позитивов":>10s} {"ридов":>8s} {"доля позитивов в классе":>24s}')
    for nm, npos, m in ((f'мёртвые (неуникален {k2}-мер)', n2, u2), (f'частично ({k1}-мер неуник., {k2} уник.)', n1 - n2, u1 & ~u2),
                        ('уникальные', n_tot - n1, ~u1)):
        print(f'{nm:34s} {npos / n_tot:9.2%} {m.mean():10.2%} {cnt[m].sum() / cnt.sum():8.2%} {m.sum() / max(npos, 1):24.5%}')
