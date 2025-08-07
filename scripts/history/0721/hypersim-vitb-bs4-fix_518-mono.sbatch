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
    --pretrained /scratch/bl3912/checkpoints/PromptDA/depth_anything_v2_metric_hypersim_vitb.pth \
    --exp-name hypersim-vitb-bs4-fix_518-rgb_fusion \
    --warm-up-steps 1000000 \
    --train-steps 200000 \
    --validate-every 1000 \
    --test-every 2500 \
    --save-step 2500 \
    --batch-size 4 \
    --ckpt-dir  /scratch/bl3912/checkpoints/PromptDA \
    --train-txt /vast/bl3912/dataset/hypersim/txt/hypersim-d_2.0_p_1.0_g_2.0-all.txt \
    --val-txt   /vast/bl3912/dataset/hypersim/txt/hypersim-d_2.0_p_1.0_g_2.0-test.txt \
                /vast/bl3912/dataset/flyingthings3d/txt/flyingThings3D-d_2.0_p_1.0_g_2.0-test.txt \
                /vast/bl3912/dataset/MIT-CGH4k/txt/MITCGH4k-d_2.0_p_1.0_g_2.0-test.txt \
    --sample-weights 1.0 \
    --test-txt  /home/bl3912/code/PromptDA/test_data/test_data.txt \
    --random-crop \
    --output-act identity \
    --mode mono \
    --downsample 0.5 \
    --input-res 518 518 