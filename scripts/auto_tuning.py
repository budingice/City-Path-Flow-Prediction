import subprocess
import os
import pandas as pd
import numpy as np
import time

# --- 1. 定义 15x15 网格寻优空间 ---
# 隐藏层维度：从 8 到 256，取 15 个等差分布的点 (约每隔 17-18 一个点)
HIDDEN_DIMS = np.linspace(8, 256, 15, dtype=int).tolist()

# 学习率：在 1e-4 (0.0001) 到 5e-2 (0.05) 之间取 15 个对数分布的点
# 对数分布能更细腻地捕捉低学习率区域的变化
LRS = np.logspace(np.log10(1e-4), np.log10(5e-2), 15).tolist()

# 结果保存路径
TARGET_CSV = "experiments/grid_search_15x15_results.csv"
SOURCE_CSV = "experiments/benchmark_summary.csv"

def run_grid_search():
    """执行 15x15 自动化网格寻优"""
    print(f"🚀 开始 15x15 深度网格寻优，共 {len(HIDDEN_DIMS) * len(LRS)} 个组合...")
    
    # 清理旧的独立结果文件，确保数据纯净
    if os.path.exists(TARGET_CSV):
        os.remove(TARGET_CSV)
        print(f"🧹 已清理旧的记录文件: {TARGET_CSV}")

    total_tasks = len(HIDDEN_DIMS) * len(LRS)
    count = 1
    
    for lr in LRS:
        for h in HIDDEN_DIMS:
            task_start = time.time()
            print(f"\n进度: [{count}/{total_tasks}] | 目标: Hidden={h}, LR={lr:.6f}")
            
            # 执行训练命令
            cmd = [
                "python", "train_entry.py",
                "--model_type", "adaptive",
                "--hidden_dim", str(h),
                "--lr", f"{lr:.6f}",
                "--exp_group", "GridSearch_15x15"
            ]
            
            try:
                # 运行实验
                subprocess.run(cmd, check=True)
                # 实时提取结果到独立 CSV
                extract_latest_result(h, lr)
            except subprocess.CalledProcessError:
                print(f"❌ 实验失败: h={h}, lr={lr:.6f}")
            
            task_end = time.time()
            print(f"⏱️ 本轮耗时: {task_end - task_start:.1f}s")
            count += 1

def extract_latest_result(h, lr):
    """从主汇总表提取当前步的实验结果并追加到 TARGET_CSV"""
    if not os.path.exists(SOURCE_CSV):
        return

    df_all = pd.read_csv(SOURCE_CSV)
    # 匹配本次实验的唯一标识
    tag = f"GridSearch_15x15_adaptive_h{h}_lr{lr:.6f}"
    
    # 提取最后一行（最新跑出来的结果）
    latest_res = df_all[df_all['Model'] == tag].tail(1)
    
    if not latest_res.empty:
        # 如果文件不存在则写表头，存在则追加
        header = not os.path.exists(TARGET_CSV)
        latest_res.to_csv(TARGET_CSV, mode='a', index=False, header=header, encoding='utf-8')
        print(f"💾 结果已同步至独立 CSV")

if __name__ == "__main__":
    total_start = time.time()
    run_grid_search()
    total_end = time.time()
    
    print(f"\n" + "="*30)
    print(f"✅ 15x15 网格寻优全部完成！")
    print(f"📂 纯净结果文件: {TARGET_CSV}")
    print(f"⏳ 总耗时: {(total_end - total_start)/3600:.2f} 小时")
    print("="*30)