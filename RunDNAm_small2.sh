#!/bin/bash
#SBATCH --job-name=Bulk_DNAmethylation
#SBATCH --account=project_2015212
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_2015212/ceren/runs/bulk2/%x-%j.out
#SBATCH --error=/scratch/project_2015212/ceren/runs/bulk2/%x-%j.err

set -euo pipefail

module load tensorflow/2.18
source /projappl/project_2015212/cavachon/envs/ceren/.venv/bin/activate

export MLFLOW_TRACKING_URI="file:///scratch/project_2015212/ceren/mlruns2"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

# Make sure key dirs exist
mkdir -p /scratch/project_2015212/ceren/runs/bulk2/embeddings
mkdir -p /scratch/project_2015212/ceren/checkpoints2

cd /projappl/project_2015212/cavachon/CAVACHON

python - << 'PY'
from cavachon.workflow import Workflow
CFG = "/projappl/project_2015212/cavachon/configs/ceren/DNAm_second.yaml"
wf = Workflow(CFG)
wf.run()
PY