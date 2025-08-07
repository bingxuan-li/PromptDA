WANDB_DIR=$SCRATCH/projects/PromptDA/wandb;
WANDB_DIR=$WANDB_DIR python train.py \
    --backbone vitb \
    --pretrained /scratch/bl3912/checkpoints/PromptDA/depth_anything_v2_metric_hypersim_vitb.pth \
    --exp-name hybrid-vitb-bs16-fix_518-mono_fusion \
    --warm-up-steps 3000 \
    --train-steps 200000 \
    --validate-every 1000 \
    --test-every 1 \
    --save-step 2500 \
    --batch-size 16 \
    --ckpt-dir  /scratch/bl3912/checkpoints/PromptDA \
    --train-txt /vast/bl3912/dataset/hypersim/txt/hypersim-d_2.0_p_1.0_g_2.0-all.txt \
                /vast/bl3912/dataset/flyingthings3d/txt/flyingThings3D-d_2.0_p_1.0_g_2.0-all.txt \
                /vast/bl3912/dataset/MIT-CGH4k/txt/MITCGH4k-d_2.0_p_1.0_g_2.0-all.txt \
    --val-txt   /vast/bl3912/dataset/hypersim/txt/hypersim-d_2.0_p_1.0_g_2.0-test.txt \
                /vast/bl3912/dataset/flyingthings3d/txt/flyingThings3D-d_2.0_p_1.0_g_2.0-test.txt \
                /vast/bl3912/dataset/MIT-CGH4k/txt/MITCGH4k-d_2.0_p_1.0_g_2.0-test.txt \
    --sample-weights 0.6 0.3 0.1 \
    --test-txt  /home/bl3912/code/PromptDA/test_data/test_data.txt \
                /home/bl3912/code/PromptDA/test_data/test_data_d2.txt \
                /vast/bl3912/dataset/txt/test_d2_p1g2_r_2_15.txt \
    --downsample 0.5 1.0 1.0 \
    --random-crop \
    --output-act identity \
    --mode mono_fusion \
    --input-res 518 518 