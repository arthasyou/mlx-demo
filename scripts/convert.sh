#!/bin/sh

HF_PATH="/Users/ancient/models/DeepSeek-R1-Distill-Qwen-14B"
MLX_PATH="$HOME/models/mlx/deepseek-r1-distill-qwen-14b"

# 激活 conda 环境
eval "$(conda shell.bash hook)"  # 确保 conda 可用
conda activate mlx || { echo "Failed to activate conda environment 'mlx'"; exit 1; }

# 执行转换
mlx_lm.convert --hf-path "$HF_PATH" --mlx-path "$MLX_PATH"
