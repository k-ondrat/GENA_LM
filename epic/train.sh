#!/usr/bin/env bash
# Запуск дообучения: bash epic/train.sh CONFIG_NAME [GPUS] [TBS] [BS]
#   bash epic/train.sh epic_csrna_pilot 6 8 1
# Конфиг ищется в downstream_tasks/expression_prediction/configs/<CONFIG_NAME>.yaml
set -eo pipefail
[[ "$CONDA_DEFAULT_ENV" == epic && -n "$GENALM_HOME" ]] || {
  echo "Сначала: source /workspace-SR003.nfs2/kondratiuk/env.sh"; exit 1; }
cd "$GENALM_HOME/GENA_LM"

CFG=$1; GPUS=${2:-6}; TBS=${3:-8}; BS=${4:-1}
export CUDA_VISIBLE_DEVICES=$GPUS
NP=$(awk -F, '{print NF}' <<< "$GPUS")
(( TBS % (BS * NP) == 0 )) || { echo "TBS=$TBS не делится на BS*NP=$((BS * NP))"; exit 1; }
GAS=$(( TBS / (BS * NP) ))
PORT=$(( 29500 + RANDOM % 500 ))
LOG="$GENALM_HOME/runs/logs/${CFG}_$(date +%m%d_%H%M).log"
mkdir -p "$(dirname "$LOG")"
echo "GPUs=$GPUS NP=$NP TBS=$TBS BS=$BS GAS=$GAS port=$PORT log=$LOG"

accelerate launch --num_processes "$NP" --main_process_port "$PORT" --mixed_precision bf16 \
  --module downstream_tasks.expression_prediction.run_expression_finetuning_final \
  --experiment_config "downstream_tasks/expression_prediction/configs/${CFG}.yaml" \
  --batch_size "$BS" --gradient_accumulation_steps "$GAS" 2>&1 | tee "$LOG"
