#!/bin/bash
cd /data/hizibu7/repos/Seraph/2d-gaussian-splatting
P=/data/hizibu7/repos/Seraph/2d-gaussian-splatting/tmp_scripts/pending_f5.txt
MAX=9
while [ -s "$P" ]; do
  QC=$(squeue -u hizibu7 -h -t R,PD | grep -v qgs_rots | wc -l)
  SL=$((MAX - QC))
  if [ $SL -gt 0 ]; then
    NEXT=$(head -n $SL "$P")
    SUCCESS=""
    for s in $NEXT; do
      if sbatch "$s" 2>&1 | grep -q "Submitted batch job"; then
        SUCCESS="$SUCCESS $s"
      fi
    done
    if [ -n "$SUCCESS" ]; then
      cp "$P" "$P.tmp"
      for s in $SUCCESS; do grep -v "^${s}$" "$P.tmp" > "$P.tmp2" && mv "$P.tmp2" "$P.tmp"; done
      mv "$P.tmp" "$P"
    fi
  fi
  sleep 90
done
