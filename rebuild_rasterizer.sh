#!/bin/bash
#SBATCH --job-name=rebuild_rast
#SBATCH --output=rebuild_rast_output.log
#SBATCH --error=rebuild_rast_error.log
#SBATCH --time=20:00:00
#SBATCH --partition=batch_ce_ugrad
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=moana-r1,moana-r2

echo "============================================"
echo "Rebuilding diff-surfel-rasterization for sm_86"
echo "Start time: $(date)"
echo "============================================"

cd /data/hizibu7/repos/Seraph/2d-gaussian-splatting

source /data/hizibu7/anaconda3/etc/profile.d/conda.sh

conda activate 2dgs

echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

# Set CUDA arch for sm_75 (2080Ti) + sm_86 (3090) + sm_80 (A100)
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"

# Reinstall from source
echo "Reinstalling diff-surfel-rasterization..."
pip install submodules/diff-surfel-rasterization/ --force-reinstall --no-deps 2>&1

echo ""
echo "Verifying..."
python -c "
import diff_surfel_rasterization
print('Module loaded OK')
print('Location:', diff_surfel_rasterization.__file__)
"

# Quick render test
echo ""
echo "Quick render test..."
python -c "
import torch
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

settings = GaussianRasterizationSettings(
    image_height=128, image_width=128,
    tanfovx=1.0, tanfovy=1.0,
    bg=torch.zeros(3, device='cuda'),
    scale_modifier=1.0,
    viewmatrix=torch.eye(4, device='cuda'),
    projmatrix=torch.eye(4, device='cuda'),
    sh_degree=0,
    campos=torch.zeros(3, device='cuda'),
    prefiltered=False, debug=False
)
rasterizer = GaussianRasterizer(raster_settings=settings)
print('Rasterizer created OK on', torch.cuda.get_device_name(0))
"

echo ""
echo "============================================"
echo "Done! End time: $(date)"
echo "============================================"
