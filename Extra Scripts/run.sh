#!/bin/bash
####### Reserve computing resources #############
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G 
#SBATCH --time=23:59:59 
#SBATCH --gpus-per-node=2
###################################################
cd /work/soghigian_lab/abdullah.zubair/rerun5/rerun6
python THEONE_modified.py
