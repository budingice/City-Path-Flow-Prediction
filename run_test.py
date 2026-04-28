import yaml
import os
import sys
import argparse
from pathlib import Path

# 确保程序能找到 src 文件夹
sys.path.append(os.getcwd())

from src.data_utils.preprocess import TrafficDataPipeline

def main():
    parser = argparse.ArgumentParser(description="Traffic Data Pipeline Test Runner")
    # 更新默认步骤，加入 6 和 7
    parser.add_argument('--steps', nargs='+', type=int, default=[1, 3, 4, 5, 6, 7],
                        help='运行阶段: 1-解析, 3-匹配, 4-去噪, 5-统计, 6-路径特征, 7-模型输入')
    
    # 增加阶段 7 专用调优参数
    parser.add_argument('--top_paths', type=int, default=50, help='选取的高频路径节点数量')
    parser.add_argument('--t_step', type=int, default=60, help='时间步长(单位:秒)')
    
    args = parser.parse_args()

    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 保持你原有的初始化方式
    pipeline = TrafficDataPipeline(config)

    # --- 执行流程 ---
    if 1 in args.steps:
        print("\n--- 阶段 1: 原始数据解析 ---")
        pipeline.step_1_parse_pneuma()

    if 3 in args.steps:
        print("\n--- 阶段 3: 路网匹配 ---")
        pipeline.step_3_map_matching()

    if 4 in args.steps:
        print("\n--- 阶段 4: 轨迹去噪与清洗 ---")
        pipeline.step_4_denoise_and_clean()

    if 5 in args.steps:
        print("\n--- 阶段 5: 流量聚合与统计分析 ---")
        pipeline.step_5_statistical_analysis()
    
    if 6 in args.steps:
        print("\n--- 阶段 6: 路径特征提取 ---")
        pipeline.step_6_extract_path_features()

    if 7 in args.steps:
        print(f"\n--- 阶段 7: 构建模型输入 (TopPaths={args.top_paths}, Step={args.t_step}s) ---")
        # 将命令行参数传递给 step_7
        pipeline.step_7_generate_model_ready_data(
            num_top_paths=args.top_paths, 
            time_step_sec=args.t_step
        )
    
    # --- 流程检查逻辑 ---
    print("\n--- 流程检查结果 ---")
    proc_dir = Path(config['path']['processed_dir'])
    
    def check_exists(pattern):
        # rglob 支持通配符递归查找
        return any(proc_dir.rglob(pattern))

    check_items = {
        1: ("*_parsed.parquet", "阶段 1 (解析)"),
        3: ("*_matched.parquet", "阶段 3 (匹配)"),
        4: ("*_clean.parquet", "阶段 4 (去噪)"),
        5: ("flow_matrix_T_N.parquet", "阶段 5 (统计)"),
        6: ("path_features/*_path_kinematics.parquet", "阶段 6 (路径特征)"),
        7: ("../model_input/st_batch_data.pt", "阶段 7 (模型输入)") # 检查相对于 processed_dir 的位置
    }

    for s in args.steps:
        if s in check_items:
            pattern, name = check_items[s]
            if check_exists(pattern):
                print(f"✅ {name} 产出文件已确认。")
            else:
                # 针对阶段 7 的特殊路径检查 (因为可能在 processed_dir 同级的 model_input 下)
                model_pt = Path("data/model_input/st_batch_data.pt")
                if s == 7 and model_pt.exists():
                    print(f"✅ {name} 产出文件已确认 (路径: {model_pt})。")
                else:
                    print(f"❌ {name} 未发现产出，请检查存储逻辑。")
    
if __name__ == "__main__":
    main()