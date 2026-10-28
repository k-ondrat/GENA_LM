"""Конфиг полного дообучения на всех видах: по блоку на каждый train-шард, по valid-блоку на вид.

    python epic/gen_config.py [--name epic_csrna_full] [--iters 0] [--tbs 32] [--repeat A=2 ...]

Берёт epic_train/<A>/intervals_full/ (make_intervals.py --full) и file_mappings_<A>.csv.
--iters 0 -> одна эпоха по train при заданном --tbs. Параметры модели и оптимизатора — как в пилоте A.
"""
import argparse, glob, json, math, os, re

H = os.environ["GENALM_HOME"]
EP = f"{H}/GENA_LM/downstream_tasks/expression_prediction"
SPECIES = ["ASM119413v2", "sSquSuc5.hap2", "Ofas_2.0", "cgigas_uk_roslin_v1", "ilPloInte3.2", "jaNemVect1.1"]

p = argparse.ArgumentParser()
p.add_argument("--name", default="epic_csrna_full")
p.add_argument("--species", nargs="+", default=SPECIES)
p.add_argument("--sub", default="intervals_full")
p.add_argument("--tbs", type=int, default=32, help="эффективный батч (для расчёта эпохи)")
p.add_argument("--iters", type=int, default=0, help="0 = одна эпоха")
p.add_argument("--lr", default="3e-5")
p.add_argument("--warmup", type=int, default=1000)
p.add_argument("--valid-interval", type=int, default=2000)
p.add_argument("--save-interval", type=int, default=5000)
p.add_argument("--repeat", nargs="*", default=[], help="апсемплинг вида: СБОРКА=k (по умолчанию 1)")
a = p.parse_args()
rep = dict((k, int(v)) for k, v in (x.split("=") for x in a.repeat))

key = lambda A: re.sub(r"[^0-9A-Za-z]", "_", A)
blocks, rows, total = [], [], 0
for A in a.species:
    d = f"{H}/epic_train/{A}/{a.sub}"
    s = json.load(open(f"{d}/summary.json"))
    shards = sorted(glob.glob(f"{d}/{A}.train.forward.s*.tsv")) or [f"{d}/{A}.train.forward.tsv"]
    for f in shards:
        tag = re.search(r"\.(s\d+)\.tsv$", f)
        tag = tag.group(1) if tag else "s00"
        blocks.append(f"""
train_dataset_{key(A)}_{tag}:
  _target_: downstream_tasks.expression_prediction.csrna_dataset.CsRNADatasetTSSWindow
  targets_path: ${{HOME_PATH}}/epic_train/{A}/file_mappings_{A}.csv
  genome: ${{HOME_PATH}}/epic_data/{A}/genome/genome.fa
  forward_intervals_path: {f.replace(H, '${HOME_PATH}')}
  reverse_intervals_path: {f.replace(H, '${HOME_PATH}').replace('.forward.', '.reverse.')}
  hash_prefix: ${{HOME_PATH}}/epic_train/cache/full/{A}
  repeat_factor: {rep.get(A, 1)}
""")
    blocks.append(f"""
valid_dataset_{key(A)}:
  _target_: downstream_tasks.expression_prediction.csrna_dataset.CsRNADatasetTSSWindow
  targets_path: ${{HOME_PATH}}/epic_train/{A}/file_mappings_{A}.csv
  genome: ${{HOME_PATH}}/epic_data/{A}/genome/genome.fa
  forward_intervals_path: ${{HOME_PATH}}/epic_train/{A}/{a.sub}/{A}.valid.forward.tsv
  reverse_intervals_path: ${{HOME_PATH}}/epic_train/{A}/{a.sub}/{A}.valid.reverse.tsv
  hash_prefix: ${{HOME_PATH}}/epic_train/cache/full/{A}
""")
    n = s["train"] * rep.get(A, 1)
    total += n
    rows.append((A, s["train"], rep.get(A, 1), n, len(shards), s["valid"], s["valid_contigs"]))

iters = a.iters or math.ceil(total / a.tbs)
head = f"""# Полное дообучение full8192 на csRNA EPIC: {len(a.species)} видов, train = всё, кроме valid-контигов.
# Сгенерировано epic/gen_config.py. train-окон с учётом repeat: {total:,}; iters {iters} при TBS {a.tbs}.
TASK_NAME: expression/{a.name}
HOME_PATH: ${{oc.env:GENALM_HOME}}
MODEL_PATH_ROOT: ${{HOME_PATH}}/runs/${{TASK_NAME}}

args_params:
  _target_: builtins.dict
  init_checkpoint: ${{HOME_PATH}}/checkpoints/full8192/full8192_best.bin
  reset_lr: True
  reset_optimizer: True
  reset_iteration: True
  resume: null
  model_cfg: ./data/configs/L12-H768-A12-V32k-preln.json
  model_cls: "downstream_tasks.expression_prediction.expression_model_final:ExpressionCounts"
  model_path: ${{MODEL_PATH_ROOT}}
  input_seq_len: 8192
  optimizer: "AdamW"
  lr: {a.lr}
  weight_decay: 1e-4
  lr_scheduler: "constant_with_warmup"
  num_warmup_steps: {a.warmup}
  clip_grad_norm: 1
  iters: {iters}
  log_interval: 100
  valid_interval: {a.valid_interval}
  save_interval: {a.save_interval}
  save_best: True
  optimize_metric: "other_loss"
  optimize_mode: "min"
  early_stopping_patience: 10000000
  data_n_workers: 8
  seed: 45
  gen_tokenizer: "AIRI-Institute/gena-lm-bert-base-t2t"
  text_tokenizer: "Qwen/Qwen3-Embedding-0.6B"

model_kwargs:          # = full8192.yaml, кроме weight (вариант A пилота)
  _target_: builtins.dict
  hf: True
  hf_model_name: ${{HOME_PATH}}/models/moderngena_large_8192
  hf_model_name_decoder: ${{HOME_PATH}}/models/decoderdp0.1
  desc_model_name: "Qwen/Qwen3-Embedding-0.6B"
  freeze_dna_encoder: false
  loss_fct:
    _target_: torch.nn.MSELoss
    reduction: 'none'
  activation:
    _target_: torch.nn.Softplus
  config:
    _target_: builtins.dict
  weight: 1.0

shared_dataset_params:
  gen_tokenizer: ${{args_params.gen_tokenizer}}
  text_tokenizer: ${{args_params.text_tokenizer}}
  text_max_seq_len: 510
  bw: bw
  num_before: 4096
  gen_max_seq_len: 8192
  token_len_for_fetch: 15
  transform_targets_bw:
    _target_: downstream_tasks.expression_prediction.expression_dataset_final.logtransform
    pseudocount: 1
  seed: ${{args_params.seed}}
"""
out = f"{EP}/configs/{a.name}.yaml"
open(out, "w").write(head + "".join(blocks))

print(f"{'сборка':22s} {'train':>9s} {'repeat':>6s} {'вклад':>9s} {'доля':>6s} {'шардов':>6s} {'valid':>6s} {'v-конт.':>7s}")
for A, n, r, nn, sh, v, vc in rows:
    print(f"{A:22s} {n:9,d} {r:6d} {nn:9,d} {nn / total:6.1%} {sh:6d} {v:6,d} {vc:7d}")
n_tr = sum(r[1] for r in rows)
print(f"\nвсего train-окон {n_tr:,} (с repeat {total:,}); блоков {len(blocks)}; iters {iters} при TBS {a.tbs}")
print(f"кэш ≈ {n_tr * 0.2 / 1024:.0f} ГБ (~200 КБ/окно) | конфиг: {out}")
