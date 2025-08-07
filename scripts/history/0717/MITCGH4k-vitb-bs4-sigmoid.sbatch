#!/bin/bash
#SBATCH --account=pr_144_tandon_priority
#SBATCH --job-name=job
#SBATCH --output=/scratch/bl3912/projects/PromptDA/logs/%j.out
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=bingxuan.li@nyu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:2

RUN_DIR=/home/bl3912/code/PromptDA;
ENV_DIR=$SCRATCH/venv/prompt-da;
WANDB_DIR=$SCRATCH/projects/PromptDA/wandb;

module purge;
module load  anaconda3/2024.02;
source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh;
conda activate $ENV_DIR/;

export PYTHONNOUSERSITE=True;
export PATH=$ENV_DIR/bin:$PATH;
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ENV_DIR/lib;

cd $RUN_DIR;
WANDB_DIR=$WANDB_DIR python train.py \
    --backbone vitb \
    --exp-name MITCGH4k-vitb-bs4-sigmoid-0717 \
    --warm-up-steps 5000 \
    --save-step 10000 \
    --validate-every 1000 \
    --batch-size 4 \
    --ckpt-dir /scratch/bl3912/checkpoints/PromptDA \
    --train-txt /vast/bl3912/dataset/MIT-CGH4k/txt/all.txt \
    --val-txt   /scratch/bl3912/dataset/hyperSim/dataset/txt/val_1k.txt \
    --normalize \
    --output-act sigmoid