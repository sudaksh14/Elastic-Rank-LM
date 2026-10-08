#!/bin/bash
# Queue the GPU smoke test, then all E4a arms behind it (afterok). Safe to run once; jobs wait for free GPUs.
cd /var/scratch/skalra/elastoSLM/repo
S=$(sbatch --parsable slurm/e4_smoke.sbatch); echo "smoke job $S"
for cfg in "F 0.8" "Fc 0.8" "J 0.8" "JKD 0.8" "F 0.5" "Fc 0.5" "J 0.5"; do set -- $cfg
  J=$(sbatch --parsable --dependency=afterok:$S --kill-on-invalid-dep=yes --export=ALL,ARM=$1,CR=$2 --job-name=E4a_$1_$2 slurm/e4a_run.sbatch); echo "E4a $1 CR=$2 -> $J"
done
squeue -u skalra
