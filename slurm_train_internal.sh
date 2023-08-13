#!/bin/bash
#SBATCH --partition=g40x  # this allows the job to run on any of the existing partitions; might need to change this in the future
#SBATCH --job-name=medium_detector
#SBATCH --nodes 1  # Change this for larger jobs
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --comment=datanet
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --exclusive

# Example usage:
# sbatch slurm_train.sh BASELINE_TYPE BASELINE_NAME


# the script assumes:
# (1) a conda environment called tng installed using tng/tools/environment.yml
export PATH="/admin/home-$USER/miniconda3/condabin:$PATH"
source /admin/home-${USER}/miniconda3/etc/profile.d/conda.sh
conda activate datacomp  # install according to tng/tools/environment.yml
# (2) that the directory /admin/home-${USER}/tng is a clone of the tng repo
cd /admin/home-${USER}/datacomp

module load openmpi
module load cuda/11.8

export MASTER_ADDR=`hostname`
export MASTER_PORT=12802
export NCCL_PROTO=simple
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_DEBUG=info

export PYTHONFAULTHANDLER=1

export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0

echo $MASTER_ADDR

# sbatch slurm_train_internal.sh detector class medium 0 s3://dcnlp-hub/datacomp_rebuttal_reshard/all/detector_class_unified/
# sbatch slurm_train_internal.sh detector count medium 0 s3://dcnlp-hub/datacomp_rebuttal_reshard/all/detector_count_unified/
# sbatch slurm_train_internal.sh detector position medium 0 s3://dcnlp-hub/datacomp_rebuttal_reshard/all/detector_position_unified/

# Change these as needed!
BASELINE_TYPE=$1 # one of "basic", "clip_threshold", "text_entities" and "clustering"
BASELINE_NAME=$2 # appropriate subfolder name
SCALE=$3
SEED=$4
DATA_PATH=$5
NUM_EVALS=8
# if [ "$SCALE" = "xlarge" ]; then
#     NUM_EVALS=200
# elif [ "$SCALE" = "large" ]; then
#     NUM_EVALS=40
# elif [ "$SCALE" = "medium" ]; then
#     NUM_EVALS=8
# elif [ "$SCALE" = "small" ]; then
#     NUM_EVALS=4
# fi

EXP_NAME="${SCALE}_${BASELINE_TYPE}_${BASELINE_NAME}_sd${SEED}_noresample"
TEMP_OUTPUT_DIR="/fsx/home-${USER}/tng_exp/baselines/results/"

FAILED_RESHARDS=0
# JOB_ID=0
# while (( $JOB_ID < $NUM_RESHARDER_NODES ))
# do
#     RESHARDER_OUTPUT="$DATA_PATH/${JOB_ID}/meta.json"
#     RUN_RESHARDER=$(aws s3 ls $RESHARDER_OUTPUT)
#     if [ -z "$RUN_RESHARDER" ]; then
#         echo "Did not find the file ${RESHARDER_OUTPUT}. Training will not run unless all resharder jobs are successful."
#         FAILED_RESHARDS=$(( $FAILED_RESHARDS + 1 ))
#     fi
#     JOB_ID=$(( $JOB_ID + 1 ))
# done
echo "Found $FAILED_RESHARDS failed reshard jobs"
echo "Data path: $DATA_PATH"

if (( $FAILED_RESHARDS > 0 )); then
    echo "Exiting"
else
    echo "Launching training"
    srun --comment=datanet --cpu_bind=v --accel-bind=gn python train.py \
        --scale ${SCALE} \
        --data_dir ${DATA_PATH} \
        --output_dir ${TEMP_OUTPUT_DIR} \
        --exp_name ${EXP_NAME} \
        --precision amp_bfloat16 \
        --num_checkpoints ${NUM_EVALS} \
        --seed ${SEED} \
        --report_to_wandb \
        --accum_freq 1 \
        --save_frequency 1
fi
