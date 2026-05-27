import subprocess
import os

# 定义要跑的所有模型组合
experiments = [
    # 对比实验
    {"model": "ha", "adj": "topo", "loss": "base"},
    {"model": "linear", "adj": "topo", "loss": "base"},
    {"model": "lstm", "adj": "topo", "loss": "base"},
    {"model": "standard_stgcn", "adj": "topo", "loss": "base"},
    {"model": "adaptive", "adj": "topo", "loss": "base"},
    # 消融实验 (Adaptive + 不同矩阵)
    {"model": "adaptive", "adj": "semantic", "loss": "base"},
    # 损失函数实验
    {"model": "adaptive", "adj": "semantic", "loss": "trend"},
    {"model": "adaptive", "adj": "semantic", "loss": "weighted"},
]

def run_exp():
    # 清理旧的汇总表
    if os.path.exists("experiments/benchmark_summary.csv"):
        os.remove("experiments/benchmark_summary.csv")
        
    for exp in experiments:
        print(f"\n▶️ 正在运行: Model={exp['model']}, Adj={exp['adj']}, Loss={exp['loss']}")
        cmd = (
            f"python train_entry.py "
            f"--model_type {exp['model']} "
            f"--adj_type {exp['adj']} "
            f"--loss_mode {exp['loss']}"
        )
        subprocess.run(cmd, shell=True)

    print("\n✅ 自动化实验全部完成！请查看 experiments/benchmark_summary.csv 获取结果表格。")

if __name__ == "__main__":
    run_exp()