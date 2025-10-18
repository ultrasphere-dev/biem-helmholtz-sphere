#!/bin/sh
#PBS -q debug-mig
#PBS -o out.log
#PBS -j oe
#PBS -m abe
#PBS -l select=1

cd biem-helmholtz-sphere
git pull
~/.local/bin/uv run biem-helmholtz-sphere jascome --backend=torch --device=cuda
~/.local/bin/uv run biem-helmholtz-sphere jascome-bempp
