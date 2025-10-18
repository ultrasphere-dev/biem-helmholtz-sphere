#!/bin/sh -login
#$ -cwd
#$ -l cpu_160=1
#$ -l h_rt=00:30:00
#$ -o out.log
#$ -j y
#$ -M 34j.95a2p@simplelogin.com
#$ -m e

cd biem-helmholtz-sphere
git pull
uv run biem-helmholtz-sphere jascome
uv run biem-helmholtz-sphere jascome-bempp
