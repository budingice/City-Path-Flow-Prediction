import yaml
import os
import sys
import argparse

# 确保程序能找到 src 文件夹
sys.path.append(os.getcwd())

from src.data_utils.preprocess import TrafficDataPipeline

def main():
    # 1. 配置命令行参数
    parser = argparse.ArgumentParser(description="Traffic Data Pipeline Test Runner")
    parser.add_argument('--steps', nargs='+', type=int, default=[1, 3, 4, 5],
                        help='要运行的阶段编号，例如: --steps 1 3 (默认运行 1,3,4,5)')
    args = parser.parse_args()

    # 2. 加载配置
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 3. 初始化流水线
    pipeline = TrafficDataPipeline(config)

    # 4. 根据参数运行对应流程
    # 注意：这里跳过了 step_2，因为 step_3 内部通常会自动调用 step_2 加载路网
    
    if 1 in args.steps:
        print("\n--- 阶段 1: 原始数据解析 ---")
        pipeline.step_1_parse_pneuma()

    if 3 in args.steps:
        print("\n--- 阶段 3: 路网匹配 ---")
        pipeline.step_3_map_matching()

    if 4 in args.steps:
        print("\n--- 阶段 4: 轨迹去噪与清洗 ---")
        # 对应你新整合进 preprocess.py 的方法
        pipeline.step_4_denoise_and_clean()

    if 5 in args.steps:
        print("\n--- 阶段 5: 流量聚合与统计分析 ---")
        # 对应你新整合进 preprocess.py 的方法
        pipeline.step_5_statistical_analysis()

    # 5. 最终检查
    print("\n--- 流程检查 ---")
    processed_files = os.listdir(config['path']['processed_dir'])
    
    check_map = {
        1: ("_parsed.parquet", "阶段 1 (解析)"),
        3: ("_matched.parquet", "阶段 3 (匹配)"),
        4: ("_clean.parquet", "阶段 4 (去噪)"),
        5: ("flow_matrix_T_N.parquet", "阶段 5 (统计)")
    }

    for s in args.steps:
        suffix, name = check_map.get(s, (None, None))
        if suffix and any(suffix in f for f in processed_files if isinstance(f, str)):
            print(f"✅ {name} 产出文件已确认。")
        elif suffix:
            print(f"⚠️  {name} 运行结束，但在目录下未发现目标文件，请检查逻辑。")

if __name__ == "__main__":
    main()