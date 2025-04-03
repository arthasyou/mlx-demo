#!/bin/sh

# 如果传入参数，则使用传入的 model 名称，否则使用默认值
MODEL="${1:-DeepSeek-R1-Distill-Qwen-14B}"
HF_PATH="/Users/ancient/models/$MODEL"
MLX_PATH="$HOME/models/mlx/$MODEL"

# 激活 conda 环境
eval "$(conda shell.bash hook)"  # 确保 conda 可用
conda activate mlx || { echo "Failed to activate conda environment 'mlx'"; exit 1; }

# 执行转换
mlx_lm.convert --hf-path "$HF_PATH" --mlx-path "$MLX_PATH"
