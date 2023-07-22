#!/bin/bash

#SBATCH -p g40n404
#SBATCH --comment "<account-name>"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -G 1
#SBATCH -J "datacomp-eval"
#SBATCH --mem 32G
#SBATCH -e "error-%j.log"
#SBATCH -o "results-%j.log"
#SBATCH --requeue


# Get options

infofolder=
track=0
model="ViT-B-32"
path="openai"
results="eval_results/"

while getopts f:t:m:p:r: arg
do
    case "$arg" in
    f)  infofolder="$OPTARG";;
    t)  track="$OPTARG";;
    m)  model="$OPTARG";;
    p)  path="$OPTARG";;
    r)  results="$OPTARG";;
    esac
done

# If not from training pipeline, generate fake info.pkl

if [ -z $infofolder ]
then
    infofolder="/tmp/datanet-eval-metadata-${SLURM_JOB_ID}/"
    mkdir -p "$infofolder"
    python -c "import pickle; f=open('${infofolder}/info.pkl','wb'); pickle.dump(dict(track=${track},model='${model}',checkpoint='${path}'),f)"
    echo "wrote to temporary info.pkl"
else
    echo "using training results from ${infofolder}"
fi

# Run evaluation

python evaluate.py \
    --train "$infofolder" \
    --output "$results"

echo "finished evaluation"
