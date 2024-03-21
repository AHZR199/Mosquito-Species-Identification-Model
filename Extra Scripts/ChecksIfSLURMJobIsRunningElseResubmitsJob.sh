#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=48:00:00
#SBATCH --output=check_and_run_%j.out

#wait 30 min?
sleep 7200

#job ID to check
JOB_ID=28340356


CHECK_INTERVAL=60

while : ; do
    if squeue -j $JOB_ID > /dev/null 2>&1; then
        echo "Job $JOB_ID is still running. Checking again in $CHECK_INTERVAL seconds."
        sleep $CHECK_INTERVAL
    else
        echo "Job $JOB_ID is not running. Executing the script."
        cd /work/soghigian_lab/abdullah.zubair/rerun5/rerun6
        ./actually_run.sh
        break
    fi
done
