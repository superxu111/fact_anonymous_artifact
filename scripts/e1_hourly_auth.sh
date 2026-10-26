#!/bin/bash
# E1 小时窗数据流：auth.txt.gz 流式按小时聚合（U 用户+ANONYMOUS，滤机器账户）
# 列: hour | pair_count(观测 user-dst 对事件数) | new_pairs(首次出现对数) | nonu(非U非ANON事件数)
cd /root/autodl-tmp/projB_hetero_graph
zcat /root/autodl-tmp/data/raw/lanl/auth.txt.gz | awk -F',' '
$2 !~ /\$$/ {
  h = int($1/3600)
  key = $2 "|" $5
  pair[h]++
  if (!(key in seen)) { seen[key]=1; newp[h]++ }
  if ($2 !~ /^U/ && $2 !~ /^ANONYMOUS/) nonu[h]++
}
END {
  for (h in pair) print h, pair[h], newp[h]+0, nonu[h]+0
}' | sort -n > results/EXP-R1-E1/hourly_auth_stream.tsv
echo DONE_hourly > results/EXP-R1-E1/hourly_done.sentinel
wc -l results/EXP-R1-E1/hourly_auth_stream.tsv
