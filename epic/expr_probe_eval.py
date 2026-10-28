import os, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
df = pd.read_csv(f'{os.environ["GENALM_HOME"]}/predictions/expr_probe.tsv', sep='\t')
df['atac_x_dinuc'] = df.atac * df.dinuc
df['cage_x_dinuc'] = df.expr_cage * df.dinuc
pos = df[df.group == 'pos']
for col in ['dinuc', 'atac', 'expr_cage', 'expr_rnaseq', 'atac_x_dinuc', 'cage_x_dinuc']:
    auc = [roc_auc_score(d.group == 'pos', d[col]) for d in (df[df.group != 'near'], df[df.group != 'rand'])]
    print(f'{col:13s} | AUC регион {auc[0]:.3f} | AUC точное место {auc[1]:.3f} | Spearman {spearmanr(pos["gt"], pos[col])[0]:.3f}')
