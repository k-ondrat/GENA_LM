import gzip, os, sys, numpy as np
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr

P = f'{os.environ["GENALM_HOME"]}/predictions'
read = lambda f: np.fromiter(gzip.open(f, 'rt'), dtype=np.float32)
a = read(f'{P}/{sys.argv[1]}.txt.gz')                     # трек модели
d = read(f'{P}/jaNemVect1.1_dinuc.txt.gz')                # динуклеотиды
g = read(f'{os.environ["GENALM_HOME"]}/epic_data/jaNemVect1.1/ground_truth/csRNA-test.txt.gz')
m = a > 0; a, d, g = a[m], d[m], g[m]; pos = g > 0       # только посчитанные позиции
for w in [0, 0.5, 1, 2, 4]:
    s = a ** w * d                                        # w = вес трека относительно динуклеотида
    print(f'w={w}: AUPRC {average_precision_score(pos, s):.4f} | Spearman {spearmanr(g[pos], s[pos])[0]:.4f}')
