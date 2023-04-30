#!/bin/bash

# Change these!
#SBATCH --partition=<partition_name>
#SBATCH --job-name=<job_name>
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --comment=<comment>
#SBATCH --open-mode=append
#SBATCH --requeue

# Example usage:
# sbatch slurm_train.sh
# Run using conda and make sure to have the conda env activated when running sbatch.


module load openmpi
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
echo go $COUNT_NODE
echo $HOSTNAMES

# Change these as needed!
DATA_PATH="/path/to/data/dir"
SCALE="small"
SEED=0
OUTPUT_DIR="/path/to/output/dir"
NUM_CHECKPOINTS=8
EXP_NAME="datacomp-scale-${SCALE}-seed${SEED}"
PRECISION="amp"  # You can also use amp_bfloat16 if supported by your hardware.

# Change comment as needed
srun --comment "<comment>" --cpu_bind=v --accel-bind=gn python train.py \
    --scale ${SCALE} \
    --data_dir ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --exp_name ${EXP_NAME} \
    --precision ${PRECISION} \
    --num_checkpoints ${NUM_CHECKPOINTS} \
    --seed ${SEED} \
    --report_to_wandb \
    --dataset_resampled \
    --accum_freq 1 
