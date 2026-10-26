#!/bin/bash
# R1 GPU worker：顺序吃队列文件，DONE 哨兵幂等跳过
Q=$1
E=/root/miniconda3/envs/euler/bin/python
cd /root/autodl-tmp/projB_hetero_graph
while IFS='|' read -r outdir extra; do
  [ -z "$outdir" ] && continue
  if [ -f "$outdir/DONE.sentinel" ]; then echo "SKIP $outdir"; continue; fi
  echo "RUN $outdir $(date +%H:%M:%S)"
  $E src/r1_train_arm.py --out "$outdir" $extra >> logs/r1workers/$(basename $outdir).log 2>&1
  if [ -f "$outdir/DONE.sentinel" ]; then echo "OK $outdir"; else echo "FAIL $outdir"; fi
done < "$Q"
echo "QUEUE_DONE $Q $(date +%H:%M:%S)"
