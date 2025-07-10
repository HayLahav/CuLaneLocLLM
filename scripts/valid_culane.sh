#!/bin/bash

# Set distributed environment variables for single-process run
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0

# Set the GPU indexes to use - default to 0
IDX=${CUDA_VISIBLE_DEVICES:-0}  

# Setup paths
data_dir=/storage/hay/culane_dataset
# Update to use the latest checkpoint you provided
model_checkpoint=/storage/hay/lanes/LocLLM/checkpoints/ckpts/culane/checkpoint-46000
output_base_dir=/storage/hay/lanes/LocLLM/evaluation_results

# Create output directory
mkdir -p ${output_base_dir}

# Define CULane categories
categories=(
    "0_normal"
    "1_crowd"
    "2_hlight"
    "3_shadow"
    "4_noline"
    "5_arrow"
    "6_curve"
    "7_cross"
    "8_night"
)

# Optional category specification from environment variable
EVAL_CATEGORY=${EVAL_CATEGORY:-"all"}

# Function to evaluate a single category
evaluate_category() {
    local category=$1
    local cat_num=${category%%_*}  # Extract the number part
    local cat_name=${category#*_}  # Extract the name part
    
    echo "======================================================"
    echo "Evaluating category: ${category}"
    echo "======================================================"
    
    # Create category-specific output directory
    output_dir=${output_base_dir}/${category}
    mkdir -p ${output_dir}
    
    # Run evaluation
    python -u utils/valid2d.py \
        --model-name ${model_checkpoint} \
        --question-file ${data_dir}/list/test_split/test${cat_num}_${cat_name}.txt \
        --gt-file ${data_dir}/list/test_split/test${cat_num}_${cat_name}.txt \
        --image-folder ${data_dir} \
        --category ${cat_name} \
        --output-dir ${output_dir} \
        --conv-format keypoint \
        --max-samples 50 \
        --batch-size 1 \
        --num-workers 2 \
        --debug True \
        2>&1 | tee ${output_dir}/eval.log
}

if [ "$EVAL_CATEGORY" == "all" ]; then
    # Evaluate all categories
    for category in "${categories[@]}"; do
        evaluate_category $category
    done
else
    # Evaluate specific category if provided
    specific_category=$EVAL_CATEGORY
    # Find the category in the list
    found=0
    for category in "${categories[@]}"; do
        if [[ $category == *"$specific_category"* ]]; then
            evaluate_category $category
            found=1
            break
        fi
    done
    
    if [ $found -eq 0 ]; then
        echo "Category '$specific_category' not found. Available categories: ${categories[*]}"
        echo "Defaulting to normal category..."
        evaluate_category "0_normal"
    fi
fi

echo "Evaluation complete. Results saved to ${output_base_dir}"