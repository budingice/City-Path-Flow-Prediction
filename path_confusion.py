import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class TrafficFlowAnalyzer:
    def __init__(self, input_dir="path_data", bin_size='3min'):
        self.input_dir = input_dir
        self.bin_size = bin_size
        self.all_data = self._load_data()

    def _load_data(self):
        """加载所有路径运动学文件"""
        files = glob.glob(os.path.join(self.input_dir, "*_path_kinematics.parquet"))
        print(f"📂 正在读取 {len(files)} 个路径文件...")
        df = pd.concat([pd.read_parquet(f) for f in files])
        # 确保时间列是 datetime 格式
        if not np.issubdtype(df['start_time'].dtype, np.datetime64):
            df['start_time'] = pd.to_datetime(df['start_time'])
        return df

    def save_daily_diagnosis(self, output_dir='figures', start_time='08:00:00', end_time='11:00:00'):
        """
        生成、显示并保存每日流量图
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"📁 已创建保存目录: {output_dir}")

        # 锁定全局 Top 1 路径
        top_path_sig = self.all_data['path_signature'].value_counts().idxmax()
        print(f"🎯 目标路径: {top_path_sig[:30]}...")

        # 提取日期
        self.all_data['date'] = self.all_data['start_time'].dt.date
        dates = sorted(self.all_data['date'].unique())

        for d in dates:
            print(f"📈 正在分析: {d}")
            # 筛选该路径、该日期的数据
            day_df = self.all_data[(self.all_data['date'] == d) & 
                                   (self.all_data['path_signature'] == top_path_sig)].copy()
            
            if day_df.empty:
                continue

            # 构建标准 3 分钟格网
            start_grid = pd.Timestamp(f"{d} {start_time}")
            end_grid = pd.Timestamp(f"{d} {end_time}")
            time_grid = pd.date_range(start=start_grid, end=end_grid, freq=self.bin_size)

            # 聚合与重采样 (关键：NaN 会导致 Matplotlib 断开连线)
            day_df['time_bin'] = day_df['start_time'].dt.floor(self.bin_size)
            flow_series = day_df.groupby('time_bin')['track_id'].count()
            flow_full = flow_series.reindex(time_grid)

            # --- 绘图逻辑 ---
            plt.figure(figsize=(12, 6))
            
            # 1. 绘制实际测量到的流量
            plt.plot(flow_full.index, flow_full.values, '-o', color='#1f77b4', 
                     markersize=5, linewidth=1.5, label='Measured Flow (3min)')

            # 2. 标记数据空洞（缺失的 15 分钟）
            missing_points = flow_full[flow_full.isna()]
            if not missing_points.empty:
                plt.plot(missing_points.index, [0] * len(missing_points), 'rx', 
                         markersize=6, label='Data Gap (Missing)')

            # 3. 美化与标注
            plt.title(f"Traffic Flow Diagnosis | {d}", fontsize=14, pad=15)
            plt.ylabel("Vehicle Count", fontsize=12)
            plt.xlabel("Time of Day", fontsize=12)
            
            # 设置横轴时间格式
            ax = plt.gca()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
            
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.legend(loc='upper right', frameon=True)
            plt.tight_layout()
            plt.gcf().autofmt_xdate()

            # --- 同时保存并显示 ---
            save_path = os.path.join(output_dir, f"flow_diag_{d}.png")
            plt.savefig(save_path, dpi=300) # 300dpi 满足论文打印需求
            print(f"💾 已保存至: {save_path}")
            plt.show() 

# --- 直接运行 ---
if __name__ == "__main__":
    # 初始化分析器
    analyzer = TrafficFlowAnalyzer(input_dir="path_data", bin_size='3min')
    # 生成并保存图片
    analyzer.save_daily_diagnosis(output_dir='figures_diagnosis')