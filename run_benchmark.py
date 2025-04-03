import itertools
import subprocess

import yaml

CONFIG_PATH = "./config/lora_config.yaml"
BASE_ITERATIONS = 10  # 跑 10 次就够做吞吐测试

batch_sizes = [4, 8, 16]
seq_lengths = [1024, 2048, 4096]

results = []


def update_config(batch_size, seq_len):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    config["batch_size"] = batch_size
    config["max_seq_length"] = seq_len
    config["iters"] = BASE_ITERATIONS
    config["steps_per_eval"] = 999999  # 禁用验证
    config["val_batches"] = 1
    config["grad_checkpoint"] = True

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f)


def run_training():
    result = subprocess.run(
        ["mlx_lm.lora", "--config", CONFIG_PATH], capture_output=True, text=True
    )
    output = result.stdout
    return output


def extract_tokens_per_sec(log_text):
    for line in log_text.splitlines():
        if "Tokens/sec" in line:
            try:
                value = float(line.strip().split("Tokens/sec")[1].split(",")[0])
                return value
            except Exception:
                pass
    return None


# 主循环
for bs, sl in itertools.product(batch_sizes, seq_lengths):
    print(f"🧪 Testing batch_size={bs}, seq_len={sl}...")
    update_config(bs, sl)
    log = run_training()
    tokens_per_sec = extract_tokens_per_sec(log)
    if tokens_per_sec:
        results.append((bs, sl, tokens_per_sec))
        print(f"✅ Tokens/sec: {tokens_per_sec}")
    else:
        print("⚠️ Failed to parse Tokens/sec")

# 排序结果
results.sort(key=lambda x: x[2], reverse=True)
print("\n📊 Benchmark Summary:")
for bs, sl, tps in results:
    print(f"batch_size={bs}, seq_len={sl} → Tokens/sec={tps}")

print(
    f"\n🏆 Best config: batch_size={results[0][0]}, seq_len={results[0][1]} → Tokens/sec={results[0][2]}"
)

import csv

# 保存到 CSV 文件
with open("benchmark_results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["batch_size", "seq_len", "tokens_per_sec"])
    writer.writerows(results)

print("📁 已将结果保存到 benchmark_results.csv")
