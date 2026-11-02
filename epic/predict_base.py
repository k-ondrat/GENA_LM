"""Инференс модели с нуклеотидной головой (CsRNABaseHead) без API gena_expression.

Окна как в predict.py: центры с шагом --step, токенизация как TSSWindowMixin (4096 токенов до центра,
до 4094 после), в трек пишется центр ±step/2. Описание эксперимента — как при обучении
(file_mappings_<A>.csv → metadata JSON → FixedDescriptionsMixin). Выход: <out>/<contig><strand>.npz с ключами
  track = предсказание по основаниям (≈ log1p счётчика), tok = токенное предсказание, lo
(геномные координаты [lo, lo+len), '-' тоже слева направо — формат predict.py).

python epic/predict_base.py --assembly A --config CONFIG.yaml --checkpoint .../pytorch_model.bin --out DIR \
    [--intervals template.train.bed.gz --contig-list valid_contigs.txt]
"""
import argparse, gzip, importlib, json, os, sys
from collections import defaultdict
from pathlib import Path

import numpy as np, pandas as pd, pysam, torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from downstream_tasks.expression_prediction.expression_dataset_final import FixedDescriptionsMixin  # noqa: E402

H = os.environ['GENALM_HOME']
p = argparse.ArgumentParser()
p.add_argument('--assembly', required=True)
p.add_argument('--config', required=True)            # yaml обучения (model_kwargs, токенизаторы, окно)
p.add_argument('--checkpoint', required=True)
p.add_argument('--out', required=True)
p.add_argument('--contig-list')
p.add_argument('--intervals', default='whitelist.test.bed.gz')
p.add_argument('--step', type=int, default=20000)
p.add_argument('--batch', type=int, default=4)
a = p.parse_args()

cfg = OmegaConf.load(a.config)
ap, sp = cfg.args_params, cfg.shared_dataset_params
NB, NMAX, TLF = int(sp.num_before), int(sp.gen_max_seq_len), int(sp.token_len_for_fetch)
NA = NMAX - NB

# модель
mod, cls = ap.model_cls.split(':')
model = getattr(importlib.import_module(mod), cls)(**instantiate(cfg.model_kwargs))
miss, unexp = model.load_state_dict(torch.load(a.checkpoint, map_location='cpu'), strict=False)
assert not [k for k in miss if 'base_head' in k], f'в чекпоинте нет головы: {miss[:5]}'
model = model.cuda().eval()

# токенизаторы и описание
gtok = AutoTokenizer.from_pretrained(ap.gen_tokenizer, trust_remote_code=True)
ttok = AutoTokenizer.from_pretrained(ap.text_tokenizer, padding_side='left')
fm_path = f'{H}/epic_train/{a.assembly}/file_mappings_{a.assembly}.csv'
row = pd.read_csv(fm_path).iloc[0]
cpath = row['metadata'] if os.path.isabs(row['metadata']) else os.path.join(os.path.dirname(fm_path), row['metadata'])
text = FixedDescriptionsMixin.make_description_from_json(json.load(open(cpath)), row['id'], cpath)
denc = ttok(text, truncation=True, max_length=int(sp.text_max_seq_len), return_tensors='pt')
print('описание:', text, flush=True)

LUT = np.full(256, 4, np.int64)
for i, ch in enumerate(b'ACGT'): LUT[ch] = LUT[ch + 32] = i
COMP = np.array([3, 2, 1, 0, 4], np.int64)
RC = str.maketrans('ACGTN', 'TGCAN')


def tokens(seq, take_last, limit):  # = TSSWindowMixin._tokens_from: (ids, длины)
    enc = gtok.encode_plus(seq, return_offsets_mapping=True)
    ids, mp = enc['input_ids'][1:-1], enc['offset_mapping'][1:-1]
    sl = slice(-limit, None) if take_last else slice(0, limit)
    ids, mp = ids[sl], mp[sl]
    lens = [(e - mp[i - 1][1] if i > 0 else e) if t == 5 else e - s for i, (t, (s, e)) in enumerate(zip(ids, mp))]
    return ids, lens


def window(fa, c, ctr, strand, L):
    """Токены и основания окна в порядке чтения + геномные границы [r0, r1)."""
    sb, sa = NB * TLF, NA * TLF
    if strand == '+':
        bi, bl = tokens(fa.fetch(c, max(ctr - sb, 0), ctr).upper(), True, NB)
        ai, al = tokens(fa.fetch(c, ctr, min(ctr + sa, L)).upper(), False, NA)
    else:
        bi, bl = tokens(fa.fetch(c, ctr, min(ctr + sb, L)).upper().translate(RC)[::-1], True, NB)
        ai, al = tokens(fa.fetch(c, max(ctr - sa, 0), ctr).upper().translate(RC)[::-1], False, NA)
    ids, lens = (bi + ai)[:NMAX - 2], np.array(bl + al)[:NMAX - 2]
    nb = sum(bl)
    r0 = ctr - nb if strand == '+' else ctr + nb - lens.sum()
    r1 = r0 + lens.sum()
    s = LUT[np.frombuffer(fa.fetch(c, r0, r1).encode(), np.uint8)]
    if strand == '-': s = COMP[s[::-1]]
    return ids, lens, s, r0, r1


@torch.no_grad()
def run(batch):
    B = len(batch)
    Lt = max(len(w[0]) for w in batch) + 2
    Lb = max(len(w[2]) for w in batch)
    ids = torch.full((B, 1, Lt), gtok.pad_token_id, dtype=torch.long)
    att = torch.zeros((B, 1, Lt), dtype=torch.long)
    bseq = torch.full((B, Lb), 4, dtype=torch.long); btok = torch.zeros((B, Lb), dtype=torch.long)
    for j, (ti, lens, s, _, _) in enumerate(batch):
        n = len(ti)
        ids[j, 0, :n + 2] = torch.tensor([gtok.cls_token_id] + list(ti) + [gtok.sep_token_id])
        att[j, 0, :n + 2] = 1
        bseq[j, :len(s)] = torch.from_numpy(s)
        btok[j, :len(s)] = torch.from_numpy(np.repeat(np.arange(1, n + 1), lens))
    kw = dict(input_ids=ids, attention_mask=att, dataset_flag=torch.ones(B, 1),
              desc_input_ids=denc['input_ids'].expand(B, -1)[:, None], desc_attention_mask=denc['attention_mask'].expand(B, -1)[:, None],
              base_seq=bseq, base_tok=btok)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(**{k: v.cuda() for k, v in kw.items()})
    tok = torch.gather(out.logits[..., 0].float(), 1, btok.cuda())
    return out.base_logits.float().cpu().numpy(), tok.cpu().numpy()


D = f'{H}/epic_data/{a.assembly}'
os.makedirs(a.out, exist_ok=True)
wl = defaultdict(list)
for l in gzip.open(f'{D}/for_participants/{a.intervals}', 'rt'):
    c, s, e = l.split()[:3]; wl[c].append((int(s), int(e)))
contigs = list(wl)
if a.contig_list:
    keep = set(open(a.contig_list).read().split())
    contigs = [c for c in contigs if c in keep]
fa = pysam.FastaFile(f'{D}/genome/genome.fa')
half = a.step // 2
print(f'контигов: {len(contigs)} | чекпоинт: {a.checkpoint}', flush=True)

for i, c in enumerate(contigs):
    lo, hi, L = min(s for s, _ in wl[c]), max(e for _, e in wl[c]), fa.get_reference_length(c)
    centers = [min(x, L - 1) for x in range(lo + half, hi + half, a.step)]
    for strand in '+-':
        out = f'{a.out}/{c}{strand}.npz'
        if os.path.exists(out): continue
        tr = {k: np.full(hi - lo, np.nan, np.float32) for k in ('track', 'tok')}
        for k in range(0, len(centers), a.batch):
            batch = [window(fa, c, ctr, strand, L) + (ctr,) for ctr in centers[k:k + a.batch]]
            P, T = run([w[:5] for w in batch])
            for j, (_, _, s, r0, r1, ctr) in enumerate(batch):
                n = r1 - r0
                g = np.arange(r0, r1) if strand == '+' else np.arange(r1 - 1, r0 - 1, -1)   # геномная позиция основания
                sel = (g >= max(ctr - half, lo)) & (g < min(ctr + half, hi))
                for key, v in zip(('track', 'tok'), (P, T)):
                    tr[key][g[sel] - lo] = v[j, :n][sel]
        np.savez(out, lo=lo, **tr)
        print(f'[{i + 1}/{len(contigs)}] {c}{strand}: {len(centers)} окон, покрыто {np.isfinite(tr["track"]).mean():.1%}', flush=True)
