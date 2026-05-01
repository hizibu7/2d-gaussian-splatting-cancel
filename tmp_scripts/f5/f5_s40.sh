#!/bin/bash
#SBATCH -J f5_s40
#SBATCH -o logs/f5_s40_%j.log
#SBATCH -e logs/f5_s40_%j.log
#SBATCH -t 02:00:00
#SBATCH -p batch_ce_ugrad
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --exclude=moana-r1,moana-r2,moana-u7,moana-u8
cd /data/hizibu7/repos/Seraph/2d-gaussian-splatting
source /data/hizibu7/anaconda3/etc/profile.d/conda.sh
conda activate 2dgs
OUT=eval/f5_scan40
rm -rf "$OUT"
python -u train.py -s data/DTU/scan40 -m "$OUT" --depth_ratio 1.0 -r 2 \
  --lambda_dist 1000 --lambda_normal 0.05 --seed 1 --port 26040 \
  --iterations 30000 --test_iterations 30000 --save_iterations 30000 \
  --log_cancel_v7 --log_cancel_v9 --log_trajectory_every 200
echo "=== $OUT contents ==="
ls -la "$OUT"/*.npz 2>/dev/null
