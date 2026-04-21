import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def check_adjacency_logic():
    # --- 1. 加载数据 ---
    data_path = "model_inputs/st_batch_data.pt"
    if not os.path.exists(data_path):
        print(f"❌ 未找到文件: {data_path}，请先运行 step5。")
        return

    data = torch.load(data_path)
    adj = data['adj']           # (50, 50) 相似度矩阵
    paths = data['path_labels']  # Top 50 路径对应的 Edge ID 列表
    
    print("="*50)
    print("🔍 邻接矩阵深度检查报告")
    print("="*50)
    # 1. 提取前10个节点的局部子矩阵
    n = 10
    sub_matrix = adj[:n, :n]

    # 转换为 DataFrame 方便查看
    df_sub = pd.DataFrame(
        sub_matrix, 
        columns=[f"P{i}" for i in range(n)], 
        index=[f"P{i}" for i in range(n)]
    )
    print(f"--- 前 {n} 条路径的相似度细节矩阵 ---")
    print(df_sub.round(2)) # 保留两位小数   
    # --- 2. 基础统计分析 ---
    num_nodes = adj.shape[0]
    # 统计非对角线的元素（即路径间的相互关系）
    off_diag = adj[~np.eye(num_nodes, dtype=bool)]
    
    print(f"1. 矩阵规模: {num_nodes} x {num_nodes}")
    print(f"2. 相似度范围: [{off_diag.min():.4f}, {off_diag.max():.4f}]")
    print(f"3. 平均相似度: {off_diag.mean():.4f}")
    print(f"4. 零相似度占比: {(off_diag == 0).mean():.2%} (即完全不重叠的路径对)")
    print(f"5. 高相似度对数 (>0.5): {np.sum(off_diag > 0.5) // 2} 对")

    # --- 3. 找出最相似的路径对 (Case Study) ---
    print("\n💡 空间关系 Case Study (Top 5 最相似路径):")
    # 找到上三角矩阵中最大的值
    tri_upper = np.triu(adj, k=1)
    top_indices = np.unravel_index(np.argsort(tri_upper.ravel())[::-1][:5], tri_upper.shape)
    
    for idx_a, idx_b in zip(*top_indices):
        sim = adj[idx_a, idx_b]
        edges_a = set(paths[idx_a])
        edges_b = set(paths[idx_b])
        overlap = edges_a & edges_b
        print(f" - Path {idx_a} & Path {idx_b}: 相似度 {sim:.2f} (共享路段: {list(overlap)})")

    # --- 4. 可视化部分 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # (A) 相似度热力图 - 观察空间聚集性
    sns.heatmap(adj, cmap='YlGnBu', ax=ax1, cbar_kws={'label': 'Jaccard Similarity'})
    ax1.set_title("Path Similarity Heatmap (Adjacency Matrix)")
    ax1.set_xlabel("Path Index")
    ax1.set_ylabel("Path Index")

    # (B) 相似度分布直方图 - 观察路网耦合程度
    ax2.hist(off_diag, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.set_title("Distribution of Path Similarities (Excluding Self)")
    ax2.set_xlabel("Jaccard Coefficient (0-1)")
    ax2.set_ylabel("Frequency")
    ax2.set_yscale('log') # 这种数据通常符合长尾分布
    
    plt.tight_layout()
    
    # 保存结果
    res_path = "ResultPicture/adj_analysis.png"
    if not os.path.exists("ResultPicture"): os.makedirs("ResultPicture")
    plt.savefig(res_path)
    print(f"\n🖼️  可视化分析图已保存至: {res_path}")
    plt.show()

if __name__ == "__main__":
    check_adjacency_logic()
