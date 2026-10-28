import argparse, gzip, json, os, numpy as np, torch, pysam
from collections import defaultdict
from gena_expression import SequenceModel, AnnotatedSequence, Feature, Condition

H = os.environ['GENALM_HOME']
p = argparse.ArgumentParser()
p.add_argument('--assembly', required=True)
p.add_argument('--condition', required=True)        # JSON с описанием эксперимента
p.add_argument('--out', required=True)
p.add_argument('--contigs', type=int, default=0)     # 0 = все контиги из --intervals
p.add_argument('--contig-list')                      # файл со списком контигов (напр. valid_contigs.txt)
p.add_argument('--intervals', default='whitelist.test.bed.gz')  # файл в for_participants/ (template.train.bed.gz для valid)
p.add_argument('--checkpoint', default=f'{H}/checkpoints/full8192/full8192_best.bin')
p.add_argument('--config', default=f'{H}/checkpoints/full8192/full8192.yaml')
p.add_argument('--step', type=int, default=20000)    # шаг центров окон = сколько п.н. берём из окна
p.add_argument('--batch', type=int, default=8)
a = p.parse_args()

D = f'{H}/epic_data/{a.assembly}'
os.makedirs(a.out, exist_ok=True)
wl = defaultdict(list)
for l in gzip.open(f'{D}/for_participants/{a.intervals}', 'rt'):
    c, s, e = l.split()[:3]; wl[c].append((int(s), int(e)))
contigs = list(wl)
if a.contig_list:
    keep = set(open(a.contig_list).read().split())
    contigs = [c for c in contigs if c in keep]
contigs = contigs[:a.contigs or None]
print(f'контигов: {len(contigs)} | чекпоинт: {a.checkpoint}', flush=True)

model = SequenceModel.load(
    model_cls=f'{H}/GENA_LM/downstream_tasks/expression_prediction/expression_model_final.py::ExpressionCounts',
    checkpoint=a.checkpoint, config=a.config,
    dna_tokenizer='AIRI-Institute/gena-lm-bert-base-t2t', description_tokenizer='Qwen/Qwen3-Embedding-0.6B',
    dna_max_seq_len=8192, desc_max_seq_len=510, token_len_for_fetch=15, num_before=4096)
cond = Condition(name=os.path.basename(a.condition), description=json.load(open(a.condition)))
fa = pysam.FastaFile(f'{D}/genome/genome.fa')
FLANK, half = 70_000, a.step // 2

for i, c in enumerate(contigs):
    lo, hi, L = min(s for s, _ in wl[c]), max(e for _, e in wl[c]), fa.get_reference_length(c)
    centers = [min(x, L - 1) for x in range(lo + half, hi + half, a.step)]
    for strand in '+-':
        out = f'{a.out}/{c}{strand}.npz'
        if os.path.exists(out): continue                 # можно перезапускать после обрыва
        track = np.full(hi - lo, np.nan, dtype=np.float32)
        for k in range(0, len(centers), 64):              # по 64 окна, чтобы не держать всё в памяти
            win = []
            for ctr in centers[k:k + 64]:
                s0, e0 = max(0, ctr - FLANK), min(L, ctr + FLANK)
                sq = AnnotatedSequence(fa.fetch(c, s0, e0).upper(), features=(Feature('tss', ctr - s0, ctr - s0 + 1, type='tss'),))
                win.append((ctr, s0, e0 - s0, sq if strand == '+' else sq.reverse_complement()))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                preds = model.predict_multiple_sequences([w[3] for w in win], condition=cond, center='tss',
                                                         max_records_per_forward=a.batch, show_progress=False)
            for (ctr, s0, n, _), pr in zip(win, preds):
                tr = pr.track('track', channel=None)
                vals = torch.as_tensor(tr.values).float().cpu().numpy()
                for tok, v in zip(tr.tokens, vals):
                    ts, te = (tok['start'], tok['end']) if strand == '+' else (n - tok['end'], n - tok['start'])
                    gs, ge = max(s0 + ts, ctr - half, lo), min(s0 + te, ctr + half, hi)
                    if ge > gs: track[gs - lo:ge - lo] = v
        np.savez(out, track=track, lo=lo)
        print(f'[{i + 1}/{len(contigs)}] {c}{strand}: {len(centers)} окон, покрыто {np.isfinite(track).mean():.1%}', flush=True)
