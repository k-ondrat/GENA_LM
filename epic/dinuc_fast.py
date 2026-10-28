import glob, gzip, os, sys, numpy as np
from pyfaidx import Fasta

A = sys.argv[1]; H = os.environ['GENALM_HOME']; D = f'{H}/epic_data/{A}'
fa = Fasta(f'{D}/genome/genome.fa', as_raw=True, sequence_always_upper=True)
code = np.full(256, 4, np.uint8); code[list(b'ACGT')] = [0, 1, 2, 3]
comp = np.array([3, 2, 1, 0, 4], np.uint8)
cache = {}
def contig(c):                       # контиг как коды 0–4 с N по краям; x[i+1] = нуклеотид i
    if c not in cache:
        cache.clear(); cache[c] = np.concatenate([[4], code[np.frombuffer(fa[c][:].encode(), np.uint8)], [4]]).astype(np.uint8)
    return cache[c]
def dn(x, s, e, st):                 # индекс динуклеотида (предыдущий, текущий) для позиций [s, e)
    return x[s:e] * 5 + x[s + 1:e + 1] if st == '+' else comp[x[s + 2:e + 2]] * 5 + comp[x[s + 1:e + 1]]
wl = lambda split: [(c, int(s), int(e)) for c, s, e in (l.split()[:3] for l in gzip.open(f'{D}/for_participants/whitelist.{split}.bed.gz', 'rt'))]

tot = np.zeros(25)
for c, s, e in wl('train'):
    x = contig(c); tot += np.bincount(dn(x, s, e, '+'), minlength=25) + np.bincount(dn(x, s, e, '-'), minlength=25)
hits = {(l.split()[0], int(l.split()[1]), l.split()[5]) for f in glob.glob(f'{D}/for_participants/csRNA*r[12]-train.bed.gz')
        for l in gzip.open(f, 'rt')}
pos = np.zeros(25)
for c, p, st in sorted(hits):
    pos[dn(contig(c), p, p + 1, st)[0]] += 1
rate = pos / np.maximum(tot, 1)
B = 'ACGTN'
print(A, 'топ динуклеотидов:', ', '.join(f'{B[i // 5]}{B[i % 5]} {rate[i]:.5f}' for i in np.argsort(-rate)[:5]))

v = rate[np.concatenate([dn(contig(c), s, e, st) for st in '+-' for c, s, e in wl('test')])].astype(np.float32)
np.save(f'{H}/predictions/{A}_dinuc.npy', v)
print('записано', len(v), 'значений')
