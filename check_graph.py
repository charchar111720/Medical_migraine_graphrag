import pandas as pd
import os

# ===================== 核心配置（改这里！）=====================
# 你的 GraphRAG 输出根目录（确保当前用户有读写权限）
OUTPUT_DIR = "/home/fang/Downloads/graphrag-main (1)/graphrag-main/output"
# 确认路径存在且有权限
assert os.path.exists(OUTPUT_DIR), f"路径不存在：{OUTPUT_DIR}"
assert os.access(OUTPUT_DIR, os.R_OK), f"无读取权限：{OUTPUT_DIR}"

# ===================== 1. 读取实体数据（nodes）=====================
nodes_path = os.path.join(OUTPUT_DIR, "entities.parquet")
if os.path.exists(nodes_path):
    nodes_df = pd.read_parquet(nodes_path)
    print("=== 抽取的实体（前5行）===")
    print(nodes_df.head())
    print(f"\n实体总数：{len(nodes_df)}")
    # 实体有 type 列，可正常统计
    if 'type' in nodes_df.columns:
        print(f"实体类型分布：\n{nodes_df['type'].value_counts()}")
    else:
        print("⚠️ 实体数据无 'type' 列")
else:
    print(f"⚠️ 实体文件不存在：{nodes_path}")

# ===================== 2. 读取关系数据（edges）=====================
edges_path = os.path.join(OUTPUT_DIR, "relationships.parquet")
if os.path.exists(edges_path):
    edges_df = pd.read_parquet(edges_path)
    print("\n=== 抽取的关系（前5行）===")
    print(edges_df.head())
    print(f"\n关系总数：{len(edges_df)}")
    # 关键：关系数据无 type 列，改为统计可用列（如 source/target）
    print("关系来源实体分布（前10）：\n", edges_df['source'].value_counts().head(10))
    print("关系目标实体分布（前10）：\n", edges_df['target'].value_counts().head(10))
else:
    print(f"⚠️ 关系文件不存在：{edges_path}")

# ===================== 3. 读取社区数据（communities）=====================
communities_path = os.path.join(OUTPUT_DIR, "communities.parquet")
if os.path.exists(communities_path):
    communities_df = pd.read_parquet(communities_path)
    print("\n=== 抽取的社区（前5行）===")
    print(communities_df.head())
    print(f"\n社区总数：{len(communities_df)}")
    # 社区数据无 type 列，统计可用列（如 community）
    if 'community' in communities_df.columns:
        print("社区ID分布（前10）：\n", communities_df['community'].value_counts().head(10))
else:
    print(f"⚠️ 社区文件不存在：{communities_path}")

# ===================== 4. 读取社区报告（community_reports）=====================
community_reports_path = os.path.join(OUTPUT_DIR, "community_reports.parquet")
if os.path.exists(community_reports_path):
    community_reports_df = pd.read_parquet(community_reports_path)
    print("\n=== 社区报告（前5行）===")
    print(community_reports_df.head())
    print(f"\n社区报告总数：{len(community_reports_df)}")
    # 社区报告无 type 列，统计可用列（如 community_id）
    if 'community_id' in community_reports_df.columns:
        print("社区报告ID分布（前10）：\n", community_reports_df['community_id'].value_counts().head(10))
else:
    print(f"⚠️ 社区报告文件不存在：{community_reports_path}")

# ===================== 可选：筛选你关心的实体（示例）=====================
# 筛选 神经调控技术 相关实体（NON_DRUG_TREATMENT 类型）
if 'nodes_df' in locals() and 'type' in nodes_df.columns:
    print("\n=== 筛选非药物治疗实体（神经调控技术）===")
    treatment_nodes = nodes_df[nodes_df['type'] == "NON_DRUG_TREATMENT"]
    print(treatment_nodes[['id', 'human_readable_id', 'title']].head(20))