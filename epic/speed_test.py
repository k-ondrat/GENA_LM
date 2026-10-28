import os, gzip, time, torch
from gena_expression import SequenceModel, AnnotatedSequence, Feature

H = os.environ['GENALM_HOME']
model = SequenceModel.load(
    model_cls=f'{H}/GENA_LM/downstream_tasks/expression_prediction/expression_model_final.py::ExpressionCounts',
    checkpoint=f'{H}/checkpoints/full8192/full8192_best.bin',
    config=f'{H}/checkpoints/full8192/full8192.yaml',
    dna_tokenizer='AIRI-Institute/gena-lm-bert-base-t2t',
    description_tokenizer='Qwen/Qwen3-Embedding-0.6B',
    dna_max_seq_len=8192, desc_max_seq_len=510, token_len_for_fetch=15, num_before=4096)

# первая хромосома Nematostella (реальная ДНК вместо случайной)
seq = []
with gzip.open(f'{H}/epic_data/jaNemVect1.1/genome/genome.fa.gz', 'rt') as f:
    next(f)
    for line in f:
        if line.startswith('>') or sum(map(len, seq)) > 1_200_000: break
        seq.append(line.strip())
chrom = ''.join(seq).upper()
seqs = [AnnotatedSequence(chrom, features=(Feature('tss', c, c + 1, type='tss'),))
        for c in range(100_000, 1_100_000, 125_000)]          # 8 окон

cond = 'assay is CAGE. tissue is heart.'
for mode in ['fp32', 'bf16']:
    torch.cuda.synchronize(); t = time.time()
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(mode == 'bf16')):
        preds = model.predict_multiple_sequences(seqs, condition=cond, center='tss',
                                                 max_records_per_forward=4, show_progress=False)
    torch.cuda.synchronize()
    tr = preds[0].track('track', channel=None)
    print(f'{mode}: {(time.time() - t) / len(seqs):.2f} с/окно | окно {tr.tokens[-1]["end"] - tr.tokens[0]["start"]} п.н. '
          f'| пик памяти {torch.cuda.max_memory_allocated() / 2**30:.1f} ГБ | expr[0] {preds[0].expression:.4f}')
