import os, random, time, torch
from gena_expression import SequenceModel, AnnotatedSequence, Feature

H = os.environ['GENALM_HOME']
model = SequenceModel.load(
    model_cls=f'{H}/GENA_LM/downstream_tasks/expression_prediction/expression_model_final.py::ExpressionCounts',
    checkpoint=f'{H}/checkpoints/full8192/full8192_best.bin',
    config=f'{H}/checkpoints/full8192/full8192.yaml',
    dna_tokenizer='AIRI-Institute/gena-lm-bert-base-t2t',
    description_tokenizer='Qwen/Qwen3-Embedding-0.6B',
    dna_max_seq_len=8192, desc_max_seq_len=510,
    token_len_for_fetch=15, num_before=4096,
)
print('загружено, GPU память:', torch.cuda.memory_allocated() // 2**30, 'ГБ')

random.seed(0)
L = 200_000
seq = AnnotatedSequence(''.join(random.choice('ACGT') for _ in range(L)),
                        features=(Feature('tss', L // 2, L // 2 + 1, type='tss'),))
t = time.time()
p = model.predict_sequence(seq, condition='assay is CAGE. tissue is heart.', center='tss')
print(f'предсказание за {time.time() - t:.1f} с, expression = {p.expression}')

tr = p.track('track', channel=None)
s, e = tr.tokens[0]['start'], tr.tokens[-1]['end']
print('logits:', tuple(p.logits.shape), '| токенов:', len(tr.values), f'| окно {e - s} п.н. ({s}..{e})')
