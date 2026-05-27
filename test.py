import torch
data = torch.load('data/model_input/st_batch_data.pt')
x_list = data['x_list']
print(f"1. 轨迹总数 (Paths): {len(x_list)}")
if len(x_list) > 0:
    lengths = [len(x) for x in x_list]
    print(f"2. 轨迹长度分布: 最小={min(lengths)}, 最大={max(lengths)}, 平均={sum(lengths)/len(lengths)}")
    valid_count = sum([1 for l in lengths if l >= 15])
    print(f"3. 能够产生样本的轨迹数 (len >= 15): {valid_count}")