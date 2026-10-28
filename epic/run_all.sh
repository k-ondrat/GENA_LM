#!/bin/bash
cd /workspace-SR003.nfs2/kondratiuk/GENA_LM
H=/workspace-SR003.nfs2/kondratiuk; P=$H/predictions
COND=$H/GENA_LM_api/downstream_tasks/expression_prediction/api/data/atacseq_generated_descriptions/ATAC_HEK293T_9d72a4.json
export OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=${GPU:-6}

for A in ${SPECIES:-jaNemVect1.1 ilPloInte3.2 cgigas_uk_roslin_v1 Ofas_2.0 sSquSuc5.hap2 ASM119413v2}; do
  echo "=== $A начало $(date '+%d.%m %H:%M')"

  # 1. трек ATAC по всем тестовым контигам (GPU)
  conda run --no-capture-output -n epic python epic/predict.py --assembly $A --condition $COND \
    --out $P/${A}_atac 2>&1 | grep -E "^\[|Error"

  # 2. динуклеотидный бейзлайн (CPU)
  conda run --no-capture-output -n epic_scoring python epic/dinuc_fast.py $A

  # 3. два варианта сабмита + проверка формата
  N=$(python3 -c "import json; print(json.load(open('$H/epic_data/genome_metrics.json'))['assemblies']['$A']['template']['test']['length'])")
  EV=$([ -d $H/epic_data/$A/ground_truth ] && echo --eval)
  for V in combo track; do
    X=$([ $V = combo ] && echo "--dinuc $P/${A}_dinuc.npy --w 1")
    conda run --no-capture-output -n epic_scoring python epic/assemble.py --assembly $A --pred $P/${A}_atac \
      $X --out $P/submit_${A}_${V}.txt.gz $EV
    echo "validate $A $V: $(cd $H/EPIC_scoring && bash ./validate $N $P/submit_${A}_${V}.txt.gz)"
  done

  echo "=== $A готово $(date '+%d.%m %H:%M')"
done
echo "=== всё готово $(date '+%d.%m %H:%M')"
