import torch
import pandas as pd

# 加载你刚刚生成的 5分钟版数据
data = torch.load("model_inputs/st_batch_data.pt")
adj = data['adj']
x_example = data['x_list'][0].squeeze(-1) # 取第一个 15min 片段

print(f"✅ 成功加载。时间步数: {x_example.shape[0]}, 路径数: {x_example.shape[1]}")

# 1. 查看前 5 分钟（即第 1 个时间步）前 5 条路径的流量
# 注意：现在的索引只有 0, 1, 2
df_view = pd.DataFrame(
    x_example[:, :5], 
    index=["0-5min", "5-10min", "10-15min"],
    columns=[f"P{i}" for i in range(5)]
)

print("\n📊 5分钟粒度流量预览 (前5条路径):")
print(df_view)

# 2. 检查邻接矩阵（前5条）
print("\n🕸️ 相似度矩阵局部 (前5x5):")
print(pd.DataFrame(adj[:5, :5]).round(2))