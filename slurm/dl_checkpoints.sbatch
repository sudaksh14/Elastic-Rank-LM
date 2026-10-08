#!/bin/bash
#SBATCH --job-name=dl_ckpt
#SBATCH --partition=fatq
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=/var/scratch/skalra/elastoSLM/runs/dl_ckpt_%j.out
set -euo pipefail
# Published Elastoformer DeiT-B ladders, public SURFdrive share (README link), via WebDAV (no password).
SHARE=BJ52qJHyWAYfwct; BASE=https://surfdrive.surf.nl/public.php/webdav
OUT=/var/scratch/skalra/elastoSLM/checkpoints
for cr in 0.4 0.6 0.8; do
  d=$OUT/deitb_cr$cr; mkdir -p "$d"
  for f in Vit_b_16_Core_Level_1 Vit_b_16_Rebuilt_Level_2 Vit_b_16_Rebuilt_Level_3 Vit_b_16_Rebuilt_Level_4 Vit_b_16_Rebuilt_Level_5 Vit_b_16_Rebuilt_Level_6; do
    echo "[$(date +%T)] CR=$cr $f"
    curl -fsS -C - --retry 5 --retry-delay 10 -u "$SHARE:" -o "$d/$f.pth" "$BASE/Deit-B_CR_$cr/$f.pth"
  done
done
( cd $OUT && sha256sum deitb_cr*/*.pth > SHA256SUMS.txt && ls -l deitb_cr*/ && du -sh . )
echo DONE
