import yaml
import os
import sys

# 解决包导入问题：确保程序能找到 src 文件夹
sys.path.append(os.getcwd())

from src.data_utils.preprocess import TrafficDataPipeline

def main():
    # 1. 加载配置
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. 初始化流水线
    pipeline = TrafficDataPipeline(config)

    # 3. 运行测试流程
    print("--- 阶段 1: 原始数据解析 ---")
    pipeline.step_1_parse_pneuma()

    print("\n--- 阶段 2: 路网匹配 ---")
    pipeline.step_3_map_matching()

    # 4. 检查产出
    print("\n--- 检查结果 ---")
    files = os.listdir(config['path']['processed_dir'])
    if any("_matched.parquet" in f for f in files):
        print("✅ 恭喜！Step 1-3 已经完全跑通。")
    else:
        print("❌ 文件夹中未发现匹配后的结果，请检查报错日志。")

if __name__ == "__main__":
    main()