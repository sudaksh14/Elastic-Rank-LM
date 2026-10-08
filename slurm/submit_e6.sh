#!/bin/bash
cd /var/scratch/skalra/elastoSLM/repo
for rc in 1.0 0.5 0.3; do J=$(sbatch --parsable --export=ALL,RCAP=$rc,CR=0.8 --job-name=E6_rc$rc slurm/e6_run.sbatch); echo "E6 rcap=$rc -> $J"; done
squeue -u skalra
