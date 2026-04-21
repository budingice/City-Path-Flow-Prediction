import yaml
import os

def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# 自动获取根目录，解决跨文件夹运行时的路径问题
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))