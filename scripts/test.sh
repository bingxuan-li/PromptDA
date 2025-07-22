# python test_train.py -b vitb -e MITCGH4k-vitb-bs4-sigmoid-0717-vitb -s 40000 -m hypersim

WANDB_DIR=$SCRATCH/projects/PromptDA/wandb python train.py \
    --backbone vitb \
    --pretrained /scratch/bl3912/checkpoints/PromptDA/depth_anything_v2_metric_hypersim_vitb.pth \
    --exp-name 0721-mixed-vitb-bs4 \
    --warm-up-steps 10 \
    --train-steps 200000 \
    --validate-every 10 \
    --test-every 100 \
    --save-step 10000 \
    --batch-size 4 \
    --ckpt-dir /scratch/bl3912/checkpoints/PromptDA \
    --train-txt /vast/bl3912/dataset/hypersim/txt/hypersim-d_2.0_p_1.0_g_2.0-all.txt \
                /vast/bl3912/dataset/flyingthings3d/txt/flyingThings3D-d_2.0_p_1.0_g_2.0-all.txt \
                /vast/bl3912/dataset/MIT-CGH4k/txt/MITCGH4k-d_2.0_p_1.0_g_2.0-all.txt \
    --val-txt   /vast/bl3912/dataset/hypersim/txt/hypersim-d_2.0_p_1.0_g_2.0-test.txt \
                /vast/bl3912/dataset/flyingthings3d/txt/flyingThings3D-d_2.0_p_1.0_g_2.0-test.txt \
                /vast/bl3912/dataset/MIT-CGH4k/txt/MITCGH4k-d_2.0_p_1.0_g_2.0-test.txt \
    --sample-weights 0.5 0.4 0.1 \
    --test-txt  /home/bl3912/code/PromptDA/test_data/test_data.txt \
    --random-crop \
    --output-act identity \
    --mode rgb_fusion \
    --input-aspect-ratio 4_3
