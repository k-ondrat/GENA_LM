import gzip, json, os, numpy as np, pandas as pd, torch, pysam
from gena_expression import SequenceModel, AnnotatedSequence, Feature, Condition

H = os.environ['GENALM_HOME']; A = 'jaNemVect1.1'; D = f'{H}/epic_data/{A}'; P = f'{H}/predictions'
rng = np.random.default_rng(0)
read = lambda f: np.fromiter(gzip.open(f, 'rt'), dtype=np.float32)

wl = [l.split()[:3] for l in gzip.open(f'{D}/for_participants/whitelist.test.bed.gz', 'rt')]
c1 = wl[0][0]
pos = np.concatenate([np.arange(int(s), int(e)) for c, s, e in wl if c == c1])
n1, ntot = len(pos), sum(int(e) - int(s) for _, s, e in wl)
gt, dn = read(f'{D}/ground_truth/csRNA-test.txt.gz'), read(f'{P}/{A}_dinuc.txt.gz')

rows = []
for k, strand in enumerate('+-'):
    idx = np.arange(n1) + k * ntot                       # строки шаблона для контига 1 на этой цепи
    g = gt[idx]
    Pi = rng.choice(np.flatnonzero(g > 0), 500, replace=False)
    Ri = rng.choice(np.flatnonzero(g == 0), 500, replace=False)
    cand = pos[Pi] + rng.choice([-1, 1], 500) * rng.integers(5, 200, 500)
    j = np.minimum(np.searchsorted(pos, cand), n1 - 1)
    Ni = j[(pos[j] == cand) & (g[j] == 0)]
    for grp, ii in [('pos', Pi), ('rand', Ri), ('near', Ni)]:
        rows += [dict(strand=strand, p=int(pos[i]), group=grp, gt=float(g[i]), dinuc=float(dn[idx[i]])) for i in ii]
df = pd.DataFrame(rows)

for s in '+-':                                           # трек ATAC HEK293T, уже посчитан
    z = np.load(f'{P}/nem_atac/{c1}{s}.npz'); m = df.strand == s
    df.loc[m, 'atac'] = z['track'][df.p[m] - int(z['lo'])]
print(df.group.value_counts().to_dict(), flush=True)

model = SequenceModel.load(
    model_cls=f'{H}/GENA_LM/downstream_tasks/expression_prediction/expression_model_final.py::ExpressionCounts',
    checkpoint=f'{H}/checkpoints/full8192/full8192_best.bin', config=f'{H}/checkpoints/full8192/full8192.yaml',
    dna_tokenizer='AIRI-Institute/gena-lm-bert-base-t2t', description_tokenizer='Qwen/Qwen3-Embedding-0.6B',
    dna_max_seq_len=8192, desc_max_seq_len=510, token_len_for_fetch=15, num_before=4096)
fa = pysam.FastaFile(f'{D}/genome/genome.fa'); L = fa.get_reference_length(c1)

def window(p, s):
    s0, e0 = max(0, p - 70_000), min(L, p + 70_000)
    sq = AnnotatedSequence(fa.fetch(c1, s0, e0).upper(), features=(Feature('tss', p - s0, p - s0 + 1, type='tss'),))
    return sq if s == '+' else sq.reverse_complement()   # для «−» TSS смотрит в другую сторону

conds = {'cage': f'{H}/descriptions/CAGE/metadata/54b5f3549aeb4ea7710894da9b9a4150.json',
         'rnaseq': f'{H}/GENA_LM_api/downstream_tasks/expression_prediction/api/data/real_descriptions/ENCFF588KDY.json'}
for name, f in conds.items():
    cond, out = Condition(name=name, description=json.load(open(f))), []
    for k in range(0, len(df), 64):
        seqs = [window(r.p, r.strand) for r in df.iloc[k:k + 64].itertuples()]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out += [pr.expression for pr in model.predict_multiple_sequences(
                seqs, condition=cond, center='tss', max_records_per_forward=8, show_progress=False, retention='scalars')]
    df[f'expr_{name}'] = out
    print(name, 'готово', flush=True)
df.to_csv(f'{P}/expr_probe.tsv', sep='\t', index=False)
