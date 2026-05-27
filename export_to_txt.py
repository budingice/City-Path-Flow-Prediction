import pandas as pd
import os

def excel_to_formatted_txt(excel_path, txt_path):
    if not os.path.exists(excel_path):
        print(f"❌ 找不到文件: {excel_path}")
        return

    # 1. 读取 Excel
    df = pd.read_excel(excel_path)

    # 2. 格式化数值（保留 4 位小数，符合学术规范）
    # 排除非数值列（比如 Model_Config）
    num_cols = df.select_dtypes(include=['float64', 'float32']).columns
    df[num_cols] = df[num_cols].round(4)

    # 3. 生成美化的文本表格
    # 使用 'github' 或 'psql' 风格，这在 txt 中看起来最整齐
    table_str = df.to_markdown(index=False, tablefmt="grid")

    # 4. 写入文件
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=== 路径流量预测消融实验指标汇总 ===\n\n")
        f.write(table_str)
        f.write("\n\n* 生成时间: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))

    print(f"✅ 文本格式表格已生成: {txt_path}")
    print("\n--- 预览内容 ---")
    print(table_str)

if __name__ == "__main__":
    # 根据你的目录结构配置路径
    base_path = os.path.dirname(os.path.abspath(__file__))
    excel_file = os.path.join(base_path, "experiments/Thesis_Plots/ablation_metrics_summary.xlsx")
    txt_file = os.path.join(base_path, "experiments/Thesis_Plots/ablation_metrics_summary.txt")

    excel_to_formatted_txt(excel_file, txt_file)