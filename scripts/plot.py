import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei'] # 支持中文
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

# 1. 加载数据
df = pd.read_csv('experiments/benchmark_summary.csv')

# 2. 模型性能对比图
metrics = ['MAE', 'RMSE', 'WAPE']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, metric in enumerate(metrics):
    sns.barplot(data=df, x='Model', y=metric, ax=axes[i], palette='viridis')
    axes[i].set_title(f'各模型 {metric} 对比', fontsize=14)
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('experiments/model_comparison.png')
plt.show()

print("✅ 对比图已保存至 experiments/model_comparison.png")