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
uvx poetry install
uvx poetry run biem-helmholtz-sphere-resonance
