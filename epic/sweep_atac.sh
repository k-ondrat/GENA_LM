#!/bin/bash
cd /workspace-SR003.nfs2/kondratiuk/GENA_LM
DIR=/workspace-SR003.nfs2/kondratiuk/GENA_LM_api/downstream_tasks/expression_prediction/api/data/atacseq_generated_descriptions
P=/workspace-SR003.nfs2/kondratiuk/predictions/sweep
LOG=$P/results.tsv; mkdir -p $P
for f in $DIR/*.json; do
  n=$(basename $f .json)
  grep -q "^$n	" $LOG 2>/dev/null && continue            # уже посчитано — пропускаем
  conda run -n epic python epic/predict.py --assembly jaNemVect1.1 --contigs 1 --condition $f --out $P/$n > /dev/null 2>&1
  r=$(conda run -n epic_scoring python epic/assemble.py --assembly jaNemVect1.1 --pred $P/$n --out $P/$n.txt.gz --eval | grep AUPRC)
  echo -e "$n\t$r" | tee -a $LOG
done
sort -t' ' -k5 -gr $LOG
