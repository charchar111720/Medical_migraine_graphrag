import pandas as pd
import os

output_dir = "./output"  # 根据你的 output 路径调整
files = [
    "entities.parquet",
    "relationships.parquet",
    "text_units.parquet",
    "communities.parquet",
    "community_reports.parquet"
]

for f in files:
    path = os.path.join(output_dir, f)
    if os.path.exists(path):
        df = pd.read_parquet(path)
        csv_path = path.replace(".parquet", ".csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"已转换: {csv_path}")
    else:
        print(f"文件不存在: {path}")