#!/bin/bash


llist=("prgan_50000_256" "stylegan1_50000_256" "stylegan3_50000_1024_resize_to_256" "stylegan2_50000_1024_resize_to_256" "Diffusion_P2_50000_256_step200" \
"ldm_50000_step200" "styleswin_50000_256")


for element in "${llist[@]}"; do
    echo $element
    python main.py \
    --real_dir  \ 
    --img_dir  \
    --experiment_type ffhq \
    --n_attr 20 \
    --n_point 100 \
    --load_from_precomputed False \
    --attr_type BLIP 
done


