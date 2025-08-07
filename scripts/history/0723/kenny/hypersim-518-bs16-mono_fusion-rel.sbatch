#!/bin/bash
#SBATCH --account=pr_173_tandon_advanced
#SBATCH --job-name=job
#SBATCH --output=/scratch/kc4906/projects/PromptDA/logs/%j.out
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=bingxuan.li@nyu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1

RUN_DIR=/home/bl3912/code/PromptDA;
ENV_DIR=/scratch/bl3912/venv/prompt-da;
WANDB_DIR=$SCRATCH/projects/PromptDA/wandb;

module purge;
module load anaconda3/2024.02;
source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh;
conda activate $ENV_DIR/;

export PYTHONNOUSERSITE=True;
export PATH=$ENV_DIR/bin:$PATH;
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ENV_DIR/lib;

cd $RUN_DIR;
WANDB_DIR=$WANDB_DIR python train.py \
    --backbone vitb \
    --pretrained /scratch/bl3912/checkpoints/depthAnything/depth_anything_v2_vitb.pth \
    --ckpt-dir  /scratch/kc4906/checkpoints/PromptDA \
    --prompt-channels 3 \
    --keep-depth \
    --d-min 0.5 \
    --d-max 1.2 \
    --train-steps 200000 \
    --log-every 10 \
    --validate-every 1000 \
    --test-every 2500 \
    --save-step 2500 \
    --warm-up-steps 2000 \
    --augment-start-steps 2000000 \
    --max-global-imbalance 0.1 \
    --max-local-imbalance 0.2 \
    --max-gs-blur 2 \
    --max-gs-kernels 10 \
    --max-p-noise 1 \
    --max-g-noise 0.05 \
    --output-act identity \
    --random-crop \
    --val-txt   /vast/bl3912/dataset/txt/test_hypersim_random_5_12.txt \
                /vast/bl3912/dataset/txt/test_flyingthings3d_random_5_12.txt \
                /vast/bl3912/dataset/txt/test_MIT-CGH4k_random_5_12.txt \
                /vast/bl3912/dataset/txt/test_nano3d-video_norm_5_12.txt \
    --test-txt  /home/bl3912/code/PromptDA/test_data/test_data.txt \
                /home/bl3912/code/PromptDA/test_data/test_data_d2.txt \
                /vast/bl3912/dataset/txt/test_d2_r_5_12.txt \
    --downsample 0.5 1.0 1.0 \
    --train-txt /vast/bl3912/dataset/txt/all_hypersim_random_5_12.txt \
                /vast/bl3912/dataset/txt/all_flyingthings3d_random_5_12.txt \
                /vast/bl3912/dataset/txt/all_MIT-CGH4k_random_5_12.txt \
                /vast/bl3912/dataset/txt/all_nano3d-video_norm_5_12.txt \
    --sample-weights 1.0 0.0 0.0 0.0 \
    --mode mono_fusion \
    --input-aspect-ratio 4_3 \
    --batch-size 4 \
    --exp-name 0723-hypersim-4_3-bs4-mono_fusion-rel

   