#!/bin/sh
#PBS -q debug-mig
#PBS -o out.log
#PBS -j oe
#PBS -m abe
#PBS -l select=1

cd biem-helmholtz-sphere-1
git pull
~/.local/bin/uv run biem_helmholtz_sphere.resonance
