#!/bin/bash
# Push the working tree to DAS-6 scratch (no .git, no PDFs). Stamps GIT_SHA (+ "-dirty" if uncommitted).
set -e
export SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-$HOME/.ssh/agent.sock}
cd "$(dirname "$0")/.."
sed -i "s/$//" slurm/*.sh slurm/*.sbatch   # Slurm rejects CRLF scripts
SHA=$(git rev-parse --short HEAD); git diff --quiet HEAD && [ -z "$(git ls-files -o --exclude-standard)" ] || SHA=$SHA-dirty
echo "$SHA" > GIT_SHA
tar czf - --exclude=.git --exclude=__pycache__ --exclude="related papers" --exclude=project --exclude="*.pyc" --exclude=outputs . \
 | ssh -o BatchMode=yes skalra@fs2.das6.science.uva.nl 'mkdir -p /var/scratch/skalra/elastoSLM/repo && tar xzf - -C /var/scratch/skalra/elastoSLM/repo && echo synced'
echo "synced $SHA"
