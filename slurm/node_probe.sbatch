#!/bin/bash
# Probe a node's memory and local storage. CPU only (no GPU requested). Exclusive: one job per node (DAS-6 rule).
#SBATCH --job-name=node_probe
#SBATCH --partition=defq
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --output=/var/scratch/skalra/elastoSLM/runs/node_probe_%j.out
echo "host: $(hostname)  cpus: $(nproc)"
echo "--- memory"; grep -E "MemTotal|MemAvailable" /proc/meminfo; echo "/dev/shm (RAM-backed):"; df -hT /dev/shm
echo "--- mounts (type: nvme/ssd = local disk, nfs = network, tmpfs = RAM)"
df -hT /local /tmp /home /var/scratch 2>/dev/null
echo "--- block devices with rotational flag (0 = SSD/NVMe, 1 = HDD)"
lsblk -d -o NAME,SIZE,ROTA,TRAN,MODEL 2>/dev/null
echo "--- /local details"; ls -ld /local 2>&1; findmnt -T /local -o TARGET,SOURCE,FSTYPE,SIZE 2>/dev/null || grep " /local " /proc/mounts
D=/local/$USER-probe; mkdir -p $D 2>/dev/null && echo "/local writable by $USER: yes" || echo "/local writable by $USER: NO"
if [ -w $D ]; then
  echo "--- /local write 2GiB (fdatasync)"; dd if=/dev/zero of=$D/t.bin bs=16M count=128 conv=fdatasync 2>&1 | tail -1
  echo "--- /local read 2GiB (cold-ish, O_DIRECT)"; dd if=$D/t.bin of=/dev/null bs=16M iflag=direct 2>&1 | tail -1
  rm -rf $D
fi
N=/var/scratch/skalra/tmp/probe_$SLURM_JOB_ID; mkdir -p $N
echo "--- NFS /var/scratch write 2GiB (fdatasync)"; dd if=/dev/zero of=$N/t.bin bs=16M count=128 conv=fdatasync 2>&1 | tail -1
echo "--- NFS small-file create latency (2000 files)"; s=$(date +%s.%N); for i in $(seq 2000); do : > $N/f$i; done; e=$(date +%s.%N); echo "NFS: $(echo "($e-$s)/2000*1000" | bc -l | cut -c1-6) ms per file"; rm -rf $N
if [ -w /local ]; then s=$(date +%s.%N); for i in $(seq 2000); do : > /local/f_${SLURM_JOB_ID}_$i; done; e=$(date +%s.%N); echo "/local: $(echo "($e-$s)/2000*1000" | bc -l | cut -c1-6) ms per file"; rm -f /local/f_${SLURM_JOB_ID}_*; fi
