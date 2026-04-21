import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def generate_ppt_visuals(data_path="model_inputs/st_batch_data.pt"):
    # 1. 加载生成的矩阵
    if not os.path.exists(data_path):
        print("❌ 找不到数据文件，请先运行 build_st_features_batch()")
        return
    
    data = torch.load(data_path)
    adj = data['adj']  # 这是你的 (50, 50) 相似度矩阵
    
    # 设置绘图风格
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # --- 视觉 1: 全局关联热力图 (Heatmap) ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(adj, cmap="YlGnBu", cbar_kws={'label': 'Jaccard 相似度'})
    plt.title("路径空间关联矩阵 (Global Adjacency Matrix)", fontsize=15)
    plt.xlabel("路径索引 (Node ID)", fontsize=12)
    plt.ylabel("路径索引 (Node ID)", fontsize=12)
    plt.savefig("analysis_results/global_adj_heatmap.png", dpi=300)
    plt.show()

    # --- 视觉 2: 局部数值对比表 (Top 8x8) ---
    # 提取前8条高频路径，这在 PPT 中展示效果最好
    top_n = 8
    subset_adj = adj[:top_n, :top_n]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # 格式化数值，保留3位小数
    cell_text = [[f"{val:.3f}" for val in row] for row in subset_adj]
    columns = [f"Path {i}" for i in range(top_n)]
    
    table = ax.table(cellText=cell_text, 
                     colLabels=columns, 
                     rowLabels=columns, 
                     loc='center', 
                     cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title(f"路径相似度数值矩阵 (前 {top_n} 条高频路径)", fontsize=14, pad=20)
    plt.savefig("analysis_results/local_adj_table.png", dpi=300)
    plt.show()

    print("✨ 视觉素材已生成：")
    print("1. analysis_results/global_adj_heatmap.png (适合展示全局结构)")
    print("2. analysis_results/local_adj_table.png (适合展示具体数值相关性)")

if __name__ == "__main__":
    generate_ppt_visuals()