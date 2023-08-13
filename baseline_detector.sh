#!/bin/bash
#SBATCH --partition=cpu128
#SBATCH --job-name=detector
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --comment=datanet
#SBATCH --open-mode=append
#SBATCH --exclusive


export PATH="/admin/home-$USER/miniconda3/condabin:$PATH"
source /admin/home-${USER}/miniconda3/etc/profile.d/conda.sh

conda activate datacomp  # install according to tng/tools/environment.yml
cd /admin/home-${USER}/datacomp

python baselines.py --metadata_dir s3://dcnlp-hub/datacomp_rebuttal_metadata --save_path tmp2 --name detector_class --num_workers 128
sleep 3
python baselines.py --metadata_dir s3://dcnlp-hub/datacomp_rebuttal_metadata --save_path tmp2 --name detector_count --num_workers 128
sleep 3
python baselines.py --metadata_dir s3://dcnlp-hub/datacomp_rebuttal_metadata --save_path tmp2 --name detector_position --num_workers 128
sleep 3