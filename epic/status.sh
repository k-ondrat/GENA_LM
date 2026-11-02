# bash epic/status.sh [СУФФИКС]   — по умолчанию _full; для нуклеотидной головы: bash epic/status.sh _bhA
P=$GENALM_HOME/predictions; SUF=${1:-_full}
echo "=== посчитано в *$SUF (контиг готов = есть + и - .npz)"
for d in $P/*$SUF; do A=$(basename $d $SUF)
  [[ $d == *_valid_* ]] && continue
  ls $d/shard*.txt >/dev/null 2>&1 && SH=$d || SH=$GENALM_HOME/shards/$A
  for s in $SH/shard*.txt; do
    n=$(wc -l < $s); ok=0
    while read c; do [[ -f "$d/$c+.npz" && -f "$d/$c-.npz" ]] && ok=$((ok+1)); done < $s
    printf "%-22s %-7s %4d / %-4d %s\n" $A $(basename $s .txt) $ok $n $([ $ok -eq $n ] && echo ГОТОВ)
  done
done
echo "=== идёт сейчас на GPU"
G=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader | while IFS=', ' read pid uuid mem _; do
  idx=$(echo "$G" | grep "$uuid" | cut -d, -f1); a=$(ps -o args= -p $pid)
  if [[ $a == *predict.py* || $a == *predict_base.py* ]]; then
    echo "GPU $idx  pid $pid  $(echo $a | grep -oE 'assembly [^ ]+|shard[0-9]+|valid_[a-zA-Z]+' | tr '\n' ' ')"
  else echo "GPU $idx  pid $pid  ${mem} MiB  не predict: ${a:0:60}"; fi
done
