#!/bin/bash
# Sanity job: module + venv + GPU visibility. Run before any real job.
#SBATCH --exclusive
#SBATCH --job-name=env_check
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=/var/scratch/skalra/elastoSLM/runs/env_check_%j.out
source /var/scratch/skalra/elastoSLM/repo/slurm/common.sh
echo "host: $(hostname)  python: $(which python)  nvcc: $(which nvcc 2>/dev/null || echo none)"
echo "CUDA_HOME=${CUDA_HOME:-unset}  LD_LIBRARY_PATH has cuda: $(echo $LD_LIBRARY_PATH | grep -o 'cuda[^:]*' | head -2 | tr '\n' ' ')"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'available',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python -c "import torch;x=torch.randn(4096,4096,device='cuda');print('matmul ok',float((x@x).sum().isfinite()))" 2>&1 | tail -1
nvidia-smi -L 2>&1 | head -1
