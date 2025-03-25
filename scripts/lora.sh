#!/bin/sh

current_path=$(pwd)

# 激活 conda 环境
eval "$(conda shell.bash hook)"  # 确保 conda 可用
conda activate mlx || { echo "Failed to activate conda environment 'mlx'"; exit 1; }

# 执行转换
mlx_lm.lora --config "$current_path/../config/lora_config.yaml"
