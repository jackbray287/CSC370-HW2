#!/bin/bash
#SBATCH --job-name=gp-experiments
#SBATCH --array=1-3                      # one job per dataset (1, 2, 3), all run at the same time
#SBATCH --cpus-per-task=1                # the GP code is single-threaded, more CPUs won't speed it up
#SBATCH --mem=32G
#SBATCH --time=04:00:00                  # max run time per job (hh:mm:ss)
#SBATCH --output=slurm_gp_%A_%a.out      # log file per job: %A = job id, %a = dataset number
# #SBATCH --partition=CHANGE_ME          # uncomment and set if your cluster requires a partition

# run from the folder you submitted from (where gp.py and data/ are)
cd "$SLURM_SUBMIT_DIR"

# load Python: use whichever line your cluster needs, delete the others
# module load python
# module load anaconda && conda activate base
# source ~/venv/bin/activate

echo "dataset $SLURM_ARRAY_TASK_ID on $(hostname), started $(date)"
python3 -u experiments.py "$SLURM_ARRAY_TASK_ID"     # -u: print progress to the log right away
echo "finished $(date)"
