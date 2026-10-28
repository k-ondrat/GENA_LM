import gzip, os, sys, numpy as np
from pyfaidx import Fasta
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr

A = sys.argv[1]; D = f'{os.environ["GENALM_HOME"]}/epic_data/{A}'
fa = Fasta(f'{D}/genome/genome.fa', as_raw=True, sequence_always_upper=True)
code = np.full(256, 4, np.uint8); code[list(b'ACGT')] = [0, 1, 2, 3]
comp = np.array([3, 2, 1, 0, 4], np.uint8)            # A<->T, C<->G, N->N

def feats(split):
    P, M, first = [], [], []
    wl = [l.split()[:3] for l in gzip.open(f'{D}/for_participants/whitelist.{split}.bed.gz', 'rt')]
    for c, s, e in wl:
        s, e, L = int(s), int(e), len(fa[c])
        seq = ('N' if s == 0 else '') + fa[c][max(s - 1, 0):min(e + 1, L)] + ('N' if e >= L else '')
        x = code[np.frombuffer(seq.encode(), np.uint8)]
        P.append(x[:-2] * 5 + x[1:-1])                    # «+»: предыдущий нуклеотид, текущий
        M.append(comp[x[2:]] * 5 + comp[x[1:-1]])         # «−»: то же на обратной цепи
        first.append(np.full(e - s, c == wl[0][0]))
    first = np.concatenate(first)
    return np.concatenate(P + M), np.concatenate([first, first])

read = lambda f: np.fromiter(gzip.open(f, 'rt'), dtype=np.float32)
ftr, _ = feats('train'); ytr = read(f'{D}/for_participants/csRNA-train.txt.gz') > 0
rate = np.bincount(ftr, ytr, 25) / np.maximum(np.bincount(ftr, minlength=25), 1)
fte, first = feats('test'); v = rate[fte] / rate.max()

out = f'{os.environ["GENALM_HOME"]}/predictions/{A}_dinuc.txt.gz'
with gzip.open(out, 'wt', compresslevel=4) as f: np.savetxt(f, v, fmt='%.5f')
print('записано', out)

gt_f = f'{D}/ground_truth/csRNA-test.txt.gz'
if os.path.exists(gt_f):
    g = read(gt_f)
    for name, m in [('первый контиг', first), ('весь тест', np.ones_like(first))]:
        gg, vv = g[m], v[m]; pos = gg > 0
        print(f'{name}: AUPRC {average_precision_score(pos, vv):.4f} (случайный {pos.mean():.4f}) | '
              f'Spearman по позитивам {spearmanr(gg[pos], vv[pos])[0]:.4f}')
