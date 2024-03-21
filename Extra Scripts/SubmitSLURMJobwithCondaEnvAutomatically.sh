#!/bin/bash

source ~/software/src/myconda
conda activate thesis

cd /work/soghigian_lab/abdullah.zubair/rerun5/rerun6

#submits the batch job to the scheduler
sbatch run.sh
