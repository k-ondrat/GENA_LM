import argparse, gzip, os, numpy as np
from collections import defaultdict

p = argparse.ArgumentParser()
p.add_argument('--assembly', required=True); p.add_argument('--pred', required=True); p.add_argument('--out', required=True)
p.add_argument('--dinuc'); p.add_argument('--w', type=float, default=1.0); p.add_argument('--eval', action='store_true'); p.add_argument('--partial', action='store_true')
a = p.parse_args()
D = f'{os.environ["GENALM_HOME"]}/epic_data/{a.assembly}'

wl = defaultdict(list)
for l in gzip.open(f'{D}/for_participants/whitelist.test.bed.gz', 'rt'):
    c, s, e = l.split()[:3]; wl[c].append((int(s), int(e)))
parts = []
for strand in '+-':
    for c, ivs in wl.items():
        f = f'{a.pred}/{c}{strand}.npz'
        if os.path.exists(f):
            z = np.load(f); t, lo = z['track'], int(z['lo']); parts += [t[s - lo:e - lo] for s, e in ivs]
        else:
            parts += [np.full(e - s, np.nan, np.float32) for s, e in ivs]
v = np.concatenate(parts); cov = np.isfinite(v)
print(f'строк {len(v)}, трек посчитан для {cov.mean():.1%}')
if a.pred and cov.mean() < 0.999 and not a.partial: raise SystemExit('ОШИБКА: трек посчитан не полностью, сабмит не записан')

v = np.log1p(np.clip(np.nan_to_num(v), 0, None)); v /= v.max() or 1
if a.dinuc:
    v = v ** a.w * np.load(a.dinuc); v /= v.max() or 1
with gzip.open(a.out, 'wt', compresslevel=4) as f:
    np.savetxt(f, v, fmt='%.5f')
print('записано', a.out)

if a.eval:
    from sklearn.metrics import average_precision_score
    from scipy.stats import spearmanr
    g = np.fromiter(gzip.open(f'{D}/ground_truth/csRNA-test.txt.gz', 'rt'), dtype=np.float32)[cov]
    x = v[cov]; pos = g > 0
    print(f'AUPRC {average_precision_score(pos, x):.4f} (случайный {pos.mean():.4f}) | Spearman {spearmanr(g[pos], x[pos])[0]:.4f}')
