import csv
import itertools
import os
import shutil
import subprocess

import yaml

# 文件路径
BASE_CONFIG_PATH = "./config/lora_config.yaml"
TEMP_CONFIG_PATH = "./config/lora_config_benchmark_temp.yaml"
BACKUP_CONFIG_PATH = "./config/lora_config.bak.yaml"
BASE_ITERATIONS = 10

# 固定 seq_len，测试不同 batch_size 和 grad_checkpoint
SEQ_LEN = 4096
batch_sizes = [2, 4]
grad_checkpoints = [False, True]

results = []

# 备份原始配置
shutil.copy(BASE_CONFIG_PATH, BACKUP_CONFIG_PATH)


def update_config(batch_size, grad_checkpoint):
    shutil.copy(BACKUP_CONFIG_PATH, TEMP_CONFIG_PATH)
    with open(TEMP_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    config["batch_size"] = batch_size
    config["max_seq_length"] = SEQ_LEN
    config["iters"] = BASE_ITERATIONS
    config["steps_per_eval"] = 999999
    config["val_batches"] = 1
    config["grad_checkpoint"] = grad_checkpoint

    with open(TEMP_CONFIG_PATH, "w") as f:
        yaml.dump(config, f)


def run_training():
    result = subprocess.run(
        ["mlx_lm.lora", "--config", TEMP_CONFIG_PATH], capture_output=True, text=True
    )
    return result.stdout


def extract_tokens_per_sec(log_text):
    for line in log_text.splitlines():
        if "Tokens/sec" in line:
            try:
                return float(line.strip().split("Tokens/sec")[1].split(",")[0])
            except Exception:
                pass
    return None


# 主循环
for gc, bs in itertools.product(grad_checkpoints, batch_sizes):
    print(f"🧪 Testing GC={gc}, batch_size={bs}, seq_len={SEQ_LEN}...")
    update_config(bs, gc)
    log = run_training()
    tokens_per_sec = extract_tokens_per_sec(log)
    if tokens_per_sec:
        results.append((gc, bs, SEQ_LEN, tokens_per_sec))
        print(f"✅ Tokens/sec: {tokens_per_sec}")
    else:
        print("⚠️ Failed to parse Tokens/sec")

# 输出结果
results.sort(key=lambda x: x[3], reverse=True)
print("\n📊 Benchmark Summary:")
for gc, bs, sl, tps in results:
    print(f"GC={gc}, batch_size={bs}, seq_len={sl} → Tokens/sec={tps:.2f}")

# 最优配置
if results:
    best = results[0]
    print(
        f"\n🏆 Best config: GC={best[0]}, batch_size={best[1]}, seq_len={best[2]} → Tokens/sec={best[3]:.2f}"
    )

# 写入 CSV
with open("benchmark_gc_batch.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["grad_checkpoint", "batch_size", "seq_len", "tokens_per_sec"])
    writer.writerows(results)

print("📁 已将结果保存到 benchmark_gc_batch.csv")

# 清理临时文件
os.remove(TEMP_CONFIG_PATH)
