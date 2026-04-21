import pandas as pd

# 查看原始数据和清洗后数据的结构
df_raw = pd.read_parquet('matched_data/20181024_d1_0830_0900_matched.parquet')
df_clean = pd.read_parquet('cleaned_data/20181024_d1_0830_0900_matched_cleaned.parquet')

print("=== 原始数据头部 ===")
print(df_raw.head(3))
print("\n=== 原始数据信息 ===")
print(df_raw.info())
print(df_raw.columns.tolist())
print("\n=== 清洗后数据头部 ===")
print(df_clean.head(3))
print("\n=== 清洗后数据信息 ===")
print(df_clean.info())
