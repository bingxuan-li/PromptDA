WANDB_DIR=$SCRATCH/projects/PromptDA/wandb;
WANDB_DIR=$WANDB_DIR python train.py \
    --backbone vitb \
    --pretrained /scratch/bl3912/checkpoints/PromptDA/depth_anything_v2_metric_hypersim_vitb.pth \
    --ckpt-dir  /scratch/bl3912/checkpoints/PromptDA \
    --prompt-channels 3 \
    --keep-depth \
    --d-min 0.5 \
    --d-max 1.2 \
    --train-steps 200000 \
    --log-every 1 \
    --validate-every 100 \
    --test-every 100 \
    --save-step 2500 \
    --warm-up-steps 1 \
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
    --sample-weights 0.8 0.16 0.03 0.01 \
    --mode mono_fusion \
    --true-mono \
    --input-aspect-ratio 4_3 \
    --batch-size 4 \
    --augment-start-steps 2 \
    --max-global-imbalance 0.1 \
    --max-local-imbalance 0.2 \
    --max-gs-blur 2 \
    --max-gs-kernels 10 \
    --max-p-noise 1 \
    --max-g-noise 0.05 \
    --exp-name 0723-test
