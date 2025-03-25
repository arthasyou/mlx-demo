#!/bin/sh

MODEL_PATH="$HOME/models/mlx/deepseek-qwen-14b"
MESSAGE="中医是什么"

# 激活 conda 环境
eval "$(conda shell.bash hook)"  # 确保 conda 可用
conda activate mlx || { echo "Failed to activate conda environment 'mlx'"; exit 1; }

# 执行转换
mlx_lm.generate --model "$MODEL_PATH" --prompt "$MESSAGE"
