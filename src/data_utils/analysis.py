import os
import pandas as pd
import logging
from tqdm import tqdm

def extract_path_volatility(df_list, time_labels):
    """
    基于 edge_id 统计唯一车辆数并分析波动性
    """
    all_snaps = []
    for df, label in zip(df_list, time_labels):
        # 统计每条边的唯一车辆数
        flow_snap = df.groupby('edge_id')['track_id'].nunique().reset_index(name='flow')
        flow_snap['time_label'] = label
        all_snaps.append(flow_snap)

    full_df = pd.concat(all_snaps, ignore_index=True)
    pivot = full_df.pivot_table(index='edge_id', columns='time_label', values='flow', fill_value=0)
    
    # 计算 CV (变异系数)
    stats = pd.DataFrame(index=pivot.index)
    stats['mean_flow'] = pivot.mean(axis=1)
    stats['std_flow'] = pivot.std(axis=1)
    stats['cv'] = stats['std_flow'] / (stats['mean_flow'] + 1e-6)
    
    return pivot, stats

def generate_heatmap(df, output_path, center=(37.977, 23.737)):
    """生成热力图 (需安装 folium)"""
    try:
        import folium
        from folium.plugins import HeatMap
        m = folium.Map(location=center, zoom_start=13, tiles='cartodbpositron')
        heat_data = df[['lat', 'lon']].values.tolist()
        HeatMap(heat_data, radius=8, blur=12).add_to(m)
        m.save(output_path)
    except ImportError:
        logging.warning("未安装 folium，跳过热力图生成")