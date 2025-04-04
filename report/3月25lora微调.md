### **LoRA 微调训练报告**

#### **1. 训练环境**

- **硬件**: Mac M4，16 核 GPU，48GB 内存
- **软件**: MLX
- **模型**: DeepSeek-Qwen-14B（MLX 格式）
- **LoRA 适配层**: 16 层
- **数据集**: `/Users/ancient/data/zy`
- **最大序列长度**: 2048
- **学习率**: 1e-5
- **批次大小**: 4
- **训练步数**: 100 iters
- **梯度检查点**: 关闭
- **LoRA 参数**:
  - **应用层**: `self_attn.q_proj`, `self_attn.v_proj`
  - **秩（rank）**: 8
  - **缩放因子（alpha）**: 16.0
  - **缩放系数（scale）**: 10.0
  - **dropout**: 0.0

---

#### **2. 训练日志分析**

##### **2.1 训练初始化**

- 预加载 DeepSeek-Qwen-14B 预训练模型。
- 载入训练数据集。
- LoRA 训练参数统计：
  - **可训练参数占比**: **0.014%** (2.097M / 14770.034M)

##### **2.2 训练过程**

- **训练迭代**: 100 iters
- **最大显存占用**: **39.199GB**
- **平均计算速度**:
  - **每秒迭代数（It/sec）**: 约 **0.09 ~ 0.10**
  - **每秒处理 tokens（Tokens/sec）**: 约 **114 ~ 128 tokens**
  - **总训练 tokens**: **126,492**

| 迭代次数 | 训练损失 (Train Loss) | 验证损失 (Val Loss) | 计算速度 (It/sec) | 处理 Tokens/sec | 累积训练 Tokens |
| -------- | --------------------- | ------------------- | ----------------- | --------------- | --------------- |
| Iter 1   | -                     | **1.928**           | -                 | -               | -               |
| Iter 10  | **1.883**             | -                   | **0.092**         | 121.440         | 13,165          |
| Iter 20  | **1.785**             | -                   | **0.101**         | 125.334         | 25,529          |
| Iter 30  | **1.634**             | -                   | **0.102**         | 128.537         | 38,188          |
| Iter 40  | **1.520**             | -                   | **0.096**         | 117.364         | 50,358          |
| Iter 50  | **1.398**             | -                   | **0.092**         | 122.110         | 63,562          |
| Iter 60  | **1.366**             | -                   | **0.102**         | 114.685         | 74,844          |
| Iter 70  | **1.198**             | -                   | **0.091**         | 122.929         | 88,405          |
| Iter 80  | **1.100**             | -                   | **0.093**         | 122.939         | 101,673         |
| Iter 90  | **1.055**             | -                   | **0.099**         | 119.490         | 113,772         |
| Iter 100 | **0.996**             | **0.939**           | **0.092**         | 116.603         | 126,492         |

##### **2.3 训练收敛趋势**

- **训练损失下降趋势**:
  - 从 **1.883** (Iter 10) 降到 **0.996** (Iter 100)
  - 下降幅度稳定，训练过程没有出现震荡
- **验证损失下降趋势**:
  - 从 **1.928** (Iter 1) 下降到 **0.939** (Iter 100)
  - 说明模型在 LoRA 微调后具备更好的泛化能力
- **训练速度**:
  - 计算速度较为稳定，在 **0.09 ~ 0.10 It/sec** 之间
  - 处理 Tokens 速率在 **114 ~ 128 Tokens/sec**
  - 训练共 **126,492 tokens**

---

#### **3. 训练成果**

- **最终模型**:
  - 适配器权重已保存：
    - `/Users/ancient/models/lora/adapters.safetensors`
    - `/Users/ancient/models/lora/0000100_adapters.safetensors`
- **训练结果**:
  - **训练损失（Train Loss）**: **0.996**
  - **最终验证损失（Val Loss）**: **0.939**
- **显存占用**:
  - 训练过程中峰值显存 **39.199GB**
- **计算效率**:
  - 训练速度约 **0.09 ~ 0.10 It/sec**
  - 处理速率 **114 ~ 128 Tokens/sec**

---

#### **4. 结论**

✅ **LoRA 微调成功**，训练过程平稳，损失逐渐下降，最终达到 **0.996（Train Loss）/ 0.939（Val Loss）**。  
✅ 训练速度在 **0.09 ~ 0.10 It/sec**，对于 Mac M4 硬件来说表现良好。  
✅ 适配器已成功保存，可用于 **推理或下游任务的微调**。  
✅ **后续优化建议**：

- 增加 **训练步数（iters）** 以获得更好的收敛效果。
- 调整 **LoRA `rank`** 以优化计算效率（如 `rank=4` 适用于显存有限场景）。
- 适当增加 **数据集规模** 以提升泛化能力。

---

#### **5. 推理测试**

**如何加载 LoRA 进行推理**：

```bash
mlx_lm.generate --model "/Users/ancient/models/mlx/deepseek-qwen-14b" \
                --adapter-path "/Users/ancient/models/lora/adapters.safetensors" \
                --prompt "请介绍 LoRA 量化" \
                --max-tokens 200
```

---

### **最终结论**

🚀 本次 LoRA 训练任务在 **Mac M4 (MLX)** 上顺利完成，训练收敛良好，最终损失值 **0.996（训练）/ 0.939（验证）**，模型已成功存储，可用于推理或进一步微调。
