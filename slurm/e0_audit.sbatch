#!/bin/bash
#SBATCH --job-name=E0_audit
#SBATCH --partition=fatq
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/var/scratch/skalra/elastoSLM/runs/E0_%j.out
set -uo pipefail
source /var/scratch/skalra/elastoSLM/repo/slurm/common.sh
RD=$(run_dir E0); echo "run dir: $RD"; exec > >(tee -a "$RD/logs/job.log") 2>&1
cp "$REPO/slurm/e0_audit.sbatch" "$RD/cmd.sh"
export OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=""
echo "== env"; python -c "import sys,torch,transformers,torch_pruning,timm;print(sys.version.split()[0],torch.__version__,transformers.__version__,torch_pruning.__version__,timm.__version__)"
python -c "import seaborn,matplotlib,wandb,fastdownload;print('aux imports ok')" 2>&1 | tail -1
echo "== network / HF"; curl -sI --max-time 10 https://huggingface.co | head -1
python -c "import pytest" 2>/dev/null || pip install -q --target $EROOT/pylibs pytest 2>&1 | tail -2
echo "== E0.1 flexrank tests (py3.10 + StrEnum shim)"
python -m pytest "$REPO/FlexRank/flexrank/tests" -q -x -p no:cacheprovider 2>&1 | tail -15 | tee "$RD/logs/flexrank_tests.txt"
echo "== E0.2 audit"
cd "$REPO/Elastoformer"
for cr in 0.4 0.6 0.8; do
  python tools/audit_pruning.py --model_name facebook/deit-base-patch16-224 --pruning_ratio $cr --pruning_steps 5 --out "$RD" 2>&1 | grep -v -E "Warning|warn" | tee "$RD/logs/audit_cr$cr.log"
done
echo "E0 DONE"
