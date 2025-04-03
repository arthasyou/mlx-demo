#!/bin/sh

MODEL_PATH="$HOME/models/mlx/Qwen2.5-14B-Instruct"
ADAPTER_PATH="$HOME/models/lora"
MESSAGE="<|im_start|>system\n你是一位经验丰富的中医医生。<|im_end|>\n<|im_start|>user\n根据患者四诊信息，如何分析病因病机？<|im_end|>\n<|im_start|>assistant"

# 激活 conda 环境
eval "$(conda shell.bash hook)"  # 确保 conda 可用
conda activate mlx || { echo "Failed to activate conda environment 'mlx'"; exit 1; }

# 执行转换
mlx_lm.generate --model "$MODEL_PATH" --adapter-path "$ADAPTER_PATH"  --prompt "$MESSAGE" --max-tokens 4096
