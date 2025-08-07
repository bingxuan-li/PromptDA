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
    --exp-name 0727_exp011-true_mono-hybrid-518-aug0.8-bs12-d_0.3_1.2-vitb \
    --backbone vitb \
    --prompt-channels 3 \
    --output-act identity \
    --mode true_mono \
    --pretrained /scratch/bl3912/checkpoints/PromptDA/depth_anything_v2_metric_hypersim_vitb.pth \
    --ckpt-dir /scratch/bl3912/checkpoints/PromptDA \
    --warm-up-steps 1000 \
    --train-steps 200000 \
    --save-every 2000 \
    --log-every 10 \
    --validate-every 1000 \
    --test-every 2000 \
    --batch-size 12 \
    --d-min 0.3 \
    --d-max 1.2 \
    --random-crop \
    --train-txt /vast/bl3912/dataset/txt/all_hypersim_random_0.3_1.2.txt /vast/bl3912/dataset/txt/all_flyingthings3d_random_0.3_1.2.txt /vast/bl3912/dataset/txt/all_MIT-CGH4k_random_0.3_1.2.txt /vast/bl3912/dataset/txt/all_nano3d-video_norm_0.3_1.2.txt \
    --sample-weights 0.7 0.15 0.1 0.05 \
    --input-res 518 518 \
    --val-txt /vast/bl3912/dataset/txt/test_hypersim_random_0.3_1.2.txt /vast/bl3912/dataset/txt/test_flyingthings3d_random_0.3_1.2.txt /vast/bl3912/dataset/txt/test_MIT-CGH4k_random_0.3_1.2.txt /vast/bl3912/dataset/txt/test_nano3d-video_norm_0.3_1.2.txt \
    --test-txt /home/bl3912/code/PromptDA/test_data/test_data.txt /home/bl3912/code/PromptDA/test_data/test_data_d2.txt /vast/bl3912/dataset/test/hypersim_0.3_1.2_10.txt /vast/bl3912/dataset/test/flyingthings3d_0.3_1.2_10.txt /vast/bl3912/dataset/test/MIT-CGH4k_0.3_1.2_10.txt /vast/bl3912/dataset/test/nano3d-video_0.3_1.2_20.txt \
    --test-downsample 0.5 1.0 1.0 1.0 1.0 1.0 \
    --test-res 840 840 840 840 768 1024 540 960 1536 1536 768 1024 \
    --num-workers 16 \
    --augment-start-steps 2000 \
    --augment-probability 0.8 \
    --max-global-imbalance 0.2 \
    --max-local-imbalance 0.3 \
    --max-gs-blur 3 \
    --max-gs-kernels 20 \
    --max-p-noise 1.0 \
    --max-g-noise 0.06