#!/bin/bash
#SBATCH --job-name=Bulk_test_separated
#SBATCH --account=project_2015212
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_2015212/ceren/runs/bulk/%x-%j.out
#SBATCH --error=/scratch/project_2015212/ceren/runs/bulk/%x-%j.err

set -euo pipefail

module load tensorflow/2.18
source /projappl/project_2015212/cavachon/envs/ceren/.venv/bin/activate

export MLFLOW_TRACKING_URI="file:///scratch/project_2015212/ceren/mlruns"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

# Make sure key dirs exist
mkdir -p /scratch/project_2015212/ceren/runs/bulk/embeddings
mkdir -p /scratch/project_2015212/ceren/checkpoints

cd /projappl/project_2015212/cavachon/CAVACHON

python - << 'PY'
from cavachon.workflow import Workflow
from cavachon.io.file_reader import FileReader
from cavachon.model import Model  

CFG = "/projappl/project_2015212/cavachon/configs/ceren/sparse_bulk_v5.yaml"
wf = Workflow(CFG)

#wf.self.setup_io()
#wf.self.setup_modality()
#wf.self.setup_sample()
#wf.self.setup_training()
#wf.self.setup_dataset()
#wf.self.setup_model()
#wf.self.setup_analysis()

wf.setup_mdata()
wf.setup_dataloader()
wf.model = Model.make(
    component_configs=wf.config.components,
    name=wf.config.model.name,
)

wf.setup_train_scheduler()
wf.model.compile()

for data in wf.dataloader.dataset.batch(10):
    wf.model.train_step(data)
    break

PY