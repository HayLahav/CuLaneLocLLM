#!/bin/bash

IDX=0  # For a single RTX A5000

export PYTHONPATH=$PYTHONPATH:./

data_dir=/storage/hay
output_dir=./checkpoints/ckpts/culane_improved

if [ -d ${output_dir} ];then
    echo "dir already exists"
else
    mkdir -p ${output_dir}
fi

if [ -d ${output_dir}/src ];then
    echo "src dir already exists"
else
    echo "save codes to src"
    mkdir -p ${output_dir}/src
    cp -r datasets ${output_dir}/src
    cp -r models ${output_dir}/src
    cp -r utils ${output_dir}/src
    cp -r scripts ${output_dir}/src
fi

# Create a list file of CULane images
culane_list_file=${output_dir}/culane_train.txt
echo "Creating CULane list file..."
python -c "
import os
import glob

base_dir = '${data_dir}/culane_dataset'
output_file = '${culane_list_file}'

count = 0
with open(output_file, 'w') as f:
    # Include more driver folders for better training
    train_folders = ['driver_161_90frame', 'driver_23_30frame', 'driver_182_30frame'] 
    
    for folder in train_folders:
        driver_path = os.path.join(base_dir, folder)
        # Loop through each video folder
        for video_dir in os.listdir(driver_path):
            video_path = os.path.join(driver_path, video_dir)
            if os.path.isdir(video_path):
                # Process files in video folder
                for file in os.listdir(video_path):
                    if file.endswith('.jpg'):
                        ann_file = os.path.join(video_path, file.replace('.jpg', '.lines.txt'))
                        if os.path.exists(ann_file):
                            # Use relative path from base_dir
                            image_path = os.path.relpath(os.path.join(video_path, file), base_dir)
                            f.write(f'{image_path}\\n')
                            count += 1

print(f'Created dataset list file: {output_file} with {count} images')
"

# Modify locllm.py to use vit_base instead of vit_large
echo "Adapting model to use vit_base instead of vit_large..."
sed -i 's/vision_model = vit_large(/vision_model = vit_base(/g' models/locllm.py

CUDA_VISIBLE_DEVICES=$IDX torchrun --nproc_per_node=1 --master_port=25003 \
    utils/train2d.py \
    --model_name_or_path /storage/hay/lanes/LocLLM/checkpoints/model_weights/vicuna-7b-v1.5 \
    --llama_path /storage/hay/lanes/LocLLM/checkpoints/model_weights/vicuna-7b-v1.5 \
    --data_path ${culane_list_file} \
    --image_folder ${data_dir}/culane_dataset \
    --dino_path /storage/hay/lanes/LocLLM/checkpoints/model_weights/dinov2_vitb14_pretrain.pth \
    --conv_format keypoint \
    --data_augmentation True \
    --tune_mm_mlp_adapter True \
    --freeze_llm False \
    --lora_llm_enable True \
    --freeze_vit False \
    --lora_vision_enable True \
    --use_elastic_loss True \
    --elastic_loss_weight 0.8 \
    --elastic_loss_alpha 1.0 \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 3 \
    --learning_rate 5e-4 \
    --weight_decay 0.05 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --dataloader_num_workers 4 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --report_to none \
    2>&1 | tee ${output_dir}/log.txt

# Restore the original file after training (in case we want to use vit_large for other tasks)
git checkout models/locllm.py