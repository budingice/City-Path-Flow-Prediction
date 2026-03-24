import torch
import pandas as pd

def export_flow_time_series():
    # 加载数据
    data = torch.load("model_inputs/st_batch_data.pt")
    # 选择第1个15分钟片段 (3, 50, 1)
    chunk_0 = data['x_list'][0].squeeze(-1) 
    num_steps, num_paths = chunk_0.shape
    # 转为 DataFrame
    df_flow = pd.DataFrame(
        chunk_0, 
        columns=[f"Path_{i}" for i in range(num_paths)],
        index=[f"T{t+1:02d}" for t in range(num_steps)]
    )
    
    # 保存为 CSV 供汇报使用
    df_flow.to_csv("ResultPicture/path_flow_time_series.csv")
    print("✅ 流量随时间变化表已导出至 ResultPicture 文件夹")
    
    # 打印预览
    print("\n--- 流量矩阵预览 ---")
    print(df_flow)
    

if __name__ == "__main__":
    export_flow_time_series()