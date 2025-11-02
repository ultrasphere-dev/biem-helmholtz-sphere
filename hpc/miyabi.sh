#!/bin/sh
#PBS -q debug-mig
#PBS -o out.log
#PBS -j oe
#PBS -m abe
#PBS -l select=1

module load cuda nvidia
cd ${PBS_O_WORKDIR}
cd biem-helmholtz-sphere
git pull
uv sync --all-extras
uv run biem-helmholtz-sphere jascome --backend=torch --device=cuda
uv run biem-helmholtz-sphere jascome-bempp
