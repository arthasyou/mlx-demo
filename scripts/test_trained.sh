#!/bin/sh

MODEL_PATH="$HOME/models/mlx/deepseek-qwen-14b"
ADAPTER_PATH="$HOME/models/lora"
MESSAGE="根据患者四诊信息，如何分析病因病机？"

# 激活 conda 环境
eval "$(conda shell.bash hook)"  # 确保 conda 可用
conda activate mlx || { echo "Failed to activate conda environment 'mlx'"; exit 1; }

# 执行转换
mlx_lm.generate --model "$MODEL_PATH" --adapter-path "$ADAPTER_PATH"  --prompt "$MESSAGE"
