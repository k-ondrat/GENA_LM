import gzip, os, sys, numpy as np, pyBigWig, pysam
H = os.environ['GENALM_HOME']; RAW = f'{H}/human_k562/raw'
G = '/workspace-SR003.nfs2/aspeedok/GENA_LM/downstream_tasks/expression_prediction/datasets/data/genomes/hg38/hg38.fa'
NAME, CHS = sys.argv[1], sys.argv[2:]                    # пример: hg38_K562_set1 chr19 chr21
D = f'{H}/epic_data/{NAME}'
for sub in ('genome', 'for_participants', 'ground_truth'): os.makedirs(f'{D}/{sub}', exist_ok=True)
for ext in ('', '.fai'):
    if not os.path.lexists(f'{D}/genome/genome.fa{ext}'): os.symlink(G + ext, f'{D}/genome/genome.fa{ext}')
fa = pysam.FastaFile(G)
bws = {s: pyBigWig.open(f'{RAW}/GSE135498_K562-csRNA-combo.ucsc{s}.ucsc.bigWig') for s in ('pos', 'neg')}
ivs, gt = {}, {'pos': [], 'neg': []}
for ch in CHS:
    s = fa.fetch(ch).upper()
    ok = (np.frombuffer(s.encode(), np.uint8) != ord('N')).astype(np.int8)
    e = np.flatnonzero(np.diff(np.r_[0, ok, 0]))
    ivs[ch] = [(a, b) for a, b in zip(e[::2], e[1::2]) if b - a >= 100]
    for st, bw in bws.items():
        t = np.abs(np.nan_to_num(bw.values(ch, 0, len(s), numpy=True))).astype(np.float32)
        gt[st] += [t[a:b] for a, b in ivs[ch]]
with gzip.open(f'{D}/for_participants/whitelist.test.bed.gz', 'wt') as f:
    f.writelines(f'{ch}\t{a}\t{b}\n' for ch in CHS for a, b in ivs[ch])
g = np.concatenate(gt['pos'] + gt['neg'])               # порядок шаблона: вся +, затем вся −
with gzip.open(f'{D}/ground_truth/csRNA-test.txt.gz', 'wt', compresslevel=4) as f:
    np.savetxt(f, g, fmt='%.2f')
print(f'{NAME}: {", ".join(CHS)} | {len(g)} строк, доля позитивов {(g > 0).mean():.5f}')
