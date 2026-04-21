import pandas as pd
import numpy as np
import torch
import os
import glob
import matplotlib.pyplot as plt

def analyze_traffic_coverage():
    # --- 配置 ---
    input_dir = "path_data"
    data_path = "model_inputs/st_batch_data.pt"
    
    # 1. 加载所有原始路径数据（用于计算总基数）
    path_files = glob.glob(os.path.join(input_dir, "*_paths.parquet"))
    if not path_files:
        print("❌ 找不到原始路径文件，请确认 step4 已运行。")
        return
    
    df_list = [pd.read_parquet(f) for f in path_files]
    df_all = pd.concat(df_list)
    df_all['path_tuple'] = df_all['edge_id'].apply(tuple)
    
    total_trips = len(df_all)
    unique_paths_count = df_all['path_tuple'].nunique()
    
    print(f"📊 --- 原始数据统计 ---")
    print(f"总行程数 (Total Trips): {total_trips}")
    print(f"唯一路径总数 (Unique Paths): {unique_paths_count}")

    # 2. 计算覆盖率曲线 (Accumulated Coverage)
    # 按频率排序所有路径
    path_counts = df_all['path_tuple'].value_counts()
    path_counts_norm = path_counts / total_trips
    cumulative_coverage = path_counts_norm.cumsum().values

    # 3. 获取当前 step5 选择的 Top 50 覆盖情况
    top_50_coverage = cumulative_coverage[49] if len(cumulative_coverage) >= 50 else cumulative_coverage[-1]
    
    print(f"\n🎯 --- 覆盖率分析 ---")
    print(f"Top 10  路径覆盖率: {cumulative_coverage[9]*100:.2f}%")
    print(f"Top 50  路径覆盖率: {top_50_coverage*100:.2f}%")
    print(f"Top 100 路径覆盖率: {cumulative_coverage[99]*100:.2f}%" if len(cumulative_coverage) >= 100 else "")

    # 4. 可视化：帕累托曲线 (Pareto Curve)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_coverage) + 1), cumulative_coverage, color='red', linewidth=2)
    plt.axvline(x=50, color='blue', linestyle='--', label='Top 50 Threshold')
    plt.axhline(y=top_50_coverage, color='green', linestyle='--', label=f'Coverage: {top_50_coverage:.2%}')
    
    plt.title("Path Traffic Coverage (Pareto Analysis)")
    plt.xlabel("Number of Paths (Ranked by Frequency)")
    plt.ylabel("Cumulative Coverage of Total Trips")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 只显示前 500 条路径的曲线，因为长尾太长了
    plt.xlim(0, min(500, unique_paths_count))
    plt.show()

    # 5. 结论建议
    if top_50_coverage < 0.5:
        print("\n💡 建议：当前 Top 50 覆盖率较低（不足 50%），说明交通流非常分散。")
        print("   你可以考虑：1. 增加 Top-K 数量；2. 检查 step4 提取逻辑是否产生了太多细碎的路径。")
    else:
        print("\n💡 结论：Top 50 路径覆盖了大部分交通流，模型具有良好的代表性。")

if __name__ == "__main__":
    analyze_traffic_coverage()