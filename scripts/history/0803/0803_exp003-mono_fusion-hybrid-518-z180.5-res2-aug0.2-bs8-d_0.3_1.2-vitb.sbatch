#!/bin/bash
#SBATCH --account=pr_144_tandon_priority
#SBATCH --job-name=job
#SBATCH --output=/scratch/bl3912/projects/PromptDA/logs/%j.out
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=bingxuan.li@nyu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1

RUN_DIR=/home/bl3912/code/PromptDA;
ENV_DIR=$SCRATCH/venv/prompt-da;
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
    --exp-name 0803_exp003-mono_fusion-hybrid-518-z180.5-res2-aug0.2-bs8-d_0.3_1.2-vitb \
    --backbone vitb \
    --prompt-channels 3 \
    --output-act identity \
    --mode mono_fusion \
    --resnet-enabled \
    --resnet-blocks-per-stage 2 \
    --pretrained /scratch/bl3912/checkpoints/PromptDA/depth_anything_v2_metric_hypersim_vitb.pth \
    --ckpt-dir /scratch/bl3912/checkpoints/PromptDA \
    --warm-up-steps 1000 \
    --train-steps 100000 \
    --save-every 2000 \
    --log-every 10 \
    --validate-every 1000 \
    --test-every 2000 \
    --batch-size 8 \
    --d-min 0.3 \
    --d-max 1.2 \
    --random-crop \
    --train-txt /vast/bl3912/dataset/txt/train/hypersim_z180.0_rand_0.3_1.2.txt /vast/bl3912/dataset/txt/train/flyingthings3d_z180.0_rand_0.3_1.2.txt /vast/bl3912/dataset/txt/train/MIT-CGH4k_z180.0_rand_0.3_1.2.txt /vast/bl3912/dataset/txt/train/nano3d-video_z180.0_norm_0.3_1.2.txt /vast/bl3912/dataset/txt/train/shapes_z180.0_0.3_1.2.txt \
    --sample-weights 0.7 0.15 0.1 0.04 0.01 \
    --input-res 518 518 \
    --val-txt /vast/bl3912/dataset/txt/test/hypersim_z180.0_rand_0.3_1.2_100.txt /vast/bl3912/dataset/txt/test/flyingthings3d_z180.0_rand_0.3_1.2_100.txt /vast/bl3912/dataset/txt/test/MIT-CGH4k_z180.0_rand_0.3_1.2_100.txt /vast/bl3912/dataset/txt/test/nano3d-video_z180.0_norm_0.3_1.2_100.txt \
    --test-txt /home/bl3912/code/PromptDA/test_data/test_data.txt /home/bl3912/code/PromptDA/test_data/test_data_d2.txt /home/bl3912/code/PromptDA/test_data/test_data_0731.txt /vast/bl3912/dataset/txt/test/hypersim_z180.0_rand_0.3_1.2_10.txt /vast/bl3912/dataset/txt/test/flyingthings3d_z180.0_rand_0.3_1.2_10.txt /vast/bl3912/dataset/txt/test/MIT-CGH4k_z180.0_rand_0.3_1.2_10.txt /vast/bl3912/dataset/txt/test/nano3d-video_z180.0_norm_0.3_1.2_30.txt /vast/bl3912/dataset/txt/test/shapes_z180.0_0.3_1.2.txt \
    --test-downsample 0.5 1.0 1.0 1.0 1.0 1.0 1.0 1.0 \
    --test-res 840 840 840 840 1000 1000 768 1024 540 960 1536 1536 768 1024 1000 1000 \
    --num-workers 16 \
    --confuse-lens-z z180.5 \
    --augment-start-steps 2000 \
    --augment-probability 0.2 \
    --max-global-imbalance 0.2 \
    --max-local-imbalance 0.4 \
    --max-gs-blur 2 \
    --max-gs-kernels 25 \
    --max-p-noise 0.8 \
    --max-g-noise 0.04