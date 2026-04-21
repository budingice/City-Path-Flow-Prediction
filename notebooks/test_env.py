import pandas as pd
import numpy as np
import torch
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt

def test_environment():
    print("="*30)
    print("开始环境测试...")
    print("="*30)

    # 1. 测试 Pandas 和 Numpy
    print(f"核心库检查:")
    print(f"- Pandas 版本: {pd.__version__}")
    print(f"- Numpy 版本: {np.__version__}")

    # 2. 测试 PyTorch
    print(f"\n深度学习框架 (PyTorch) 检查:")
    print(f"- PyTorch 版本: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"- GPU 是否可用: {cuda_available}")
    if cuda_available:
        print(f"- 当前 GPU 设备: {torch.cuda.get_device_name(0)}")

    # 3. 测试 OSMnx 和 NetworkX (这是你毕设最关键的地理处理库)
    print(f"\n地图处理库 (OSMnx) 检查:")
    try:
        print(f"- NetworkX 版本: {nx.__version__}")
        # 尝试下载一个极小范围的雅典路网（验证网络连接和库功能）
        # 坐标为雅典市中心的一个点
        print("- 正在尝试获取雅典局部路网图（验证库依赖性）...")
        # 缩小范围以加快测试速度
        G = ox.graph_from_point((37.9838, 23.7275), dist=300, network_type='drive')
        print(f"- 成功！路网节点数: {len(G.nodes)}, 路段数: {len(G.edges)}")
        
        # 绘制简易图（如果没有图形界面环境，这一步可能会报错，但不影响库的可用性）
        print("- 尝试绘图...")
        ox.plot_graph(G, show=False, close=False)
        print("- 绘图功能正常！")
    except Exception as e:
        print(f" 地理库测试失败，原因: {e}")

    print("\n" + "="*30)
    print("环境测试结束！")
    print("="*30)

if __name__ == "__main__":
    test_environment()