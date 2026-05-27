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
    # 步骤选择
    parser.add_argument('--steps', nargs='+', type=int, default=[1, 3, 4, 5, 6, 7],
                        help='运行阶段: 1-解析, 3-匹配, 4-去噪, 5-统计, 6-路径特征, 7-模型输入')
    
    # 命令行参数：用于覆盖 config.yaml 中的默认预处理设置
    parser.add_argument('--top_paths', type=int, default=None, help='选取的高频路径节点数量')
    parser.add_argument('--t_step', type=int, default=None, help='时间步长(单位:秒)')
    
    args = parser.parse_args()

    # 1. 加载基础配置
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 2. 动态覆盖配置 (如果命令行指定了参数，则以命令行优先)
    if args.top_paths is not None:
        config['preprocess']['num_top_paths'] = args.top_paths
    if args.t_step is not None:
        config['preprocess']['time_step_sec'] = args.t_step

    # 3. 初始化 Pipeline (现在内部的 step_7 会直接读取更新后的 config)
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
        print(f"\n--- 阶段 7: 构建模型输入 (Masked 双通道模式) ---")
        print(f"📊 当前参数: TopPaths={config['preprocess']['num_top_paths']}, Step={config['preprocess']['time_step_sec']}s")
        # 直接调用，不再传参，内部会从 self.cfg 获取
        pipeline.step_7_generate_model_ready_data()
    
    # --- 流程检查逻辑 ---
    print("\n" + "="*30)
    print("📋 流程产出检查列表")
    print("="*30)
    
    # 检查逻辑保持不变，但更新了路径提示
    model_pt = Path(config['path']['model_input_pt'])
    proc_dir = Path(config['path']['processed_dir'])
    
    check_items = {
        1: ("*_parsed.parquet", "阶段 1 (解析)"),
        3: ("*_matched.parquet", "阶段 3 (匹配)"),
        4: ("cleaned/*_clean.parquet", "阶段 4 (去噪)"),
        5: ("flow_matrix_T_N.parquet", "阶段 5 (统计)"),
        6: ("path_features/*_path_kinematics.parquet", "阶段 6 (路径特征)"),
        7: (model_pt, "阶段 7 (模型输入 .pt)")
    }

    for s in args.steps:
        if s in check_items:
            pattern, name = check_items[s]
            # 针对阶段 7 (Path对象) 和其他阶段 (rglob模式) 的统一检查
            exists = model_pt.exists() if s == 7 else any(proc_dir.rglob(pattern))
            
            if exists:
                print(f"✅ {name.ljust(15)}: [已确认]")
            else:
                print(f"❌ {name.ljust(15)}: [未发现产出]")

if __name__ == "__main__":
    main()