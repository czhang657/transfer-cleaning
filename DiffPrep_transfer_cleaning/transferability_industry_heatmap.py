import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === 路径配置 ===
csv_path = "transferability_scores.csv"
industry_info_path = "stock_industry_info.csv"
output_folder = "transferability_score"
os.makedirs(output_folder, exist_ok=True)

# === 加载数据 ===
df = pd.read_csv(csv_path)  # expects columns: source, target, rmse_source, rmse_target, score
industry_df = pd.read_csv(industry_info_path, header=None, names=["Stock", "Industry"])
industry_map = dict(zip(industry_df["Stock"], industry_df["Industry"]))

# === 映射行业分类 ===
df["Industry_i"] = df["source"].map(industry_map)
df["Industry_j"] = df["target"].map(industry_map)

# === 丢弃 industry 信息缺失的数据 ===
df = df.dropna(subset=["Industry_i", "Industry_j"])

# === 添加对角线（Industry_i == Industry_j）为 1.0 ===
industries = sorted(set(df["Industry_i"]).union(set(df["Industry_j"])))
diag_df = pd.DataFrame({
    "Industry_i": industries,
    "Industry_j": industries,
    "Transferability Score": [1.0] * len(industries)
})

df = pd.concat([
    df[["Industry_i", "Industry_j", "score"]].rename(columns={"score": "Transferability Score"}),
    diag_df
], ignore_index=True)

# === 构造 pivot 矩阵（重复格子取平均） ===
pivot_df = df.pivot_table(
    index="Industry_j",
    columns="Industry_i",
    values="Transferability Score",
    aggfunc="mean"
)

# === 删除全 NaN 的行列 ===
pivot_df = pivot_df.dropna(how="all", axis=0)
pivot_df = pivot_df.dropna(how="all", axis=1)

# === 对齐行列顺序 ===
pivot_df = pivot_df.loc[sorted(pivot_df.index), sorted(pivot_df.columns)]

# === 创建行业名到 index 的映射 ===
industry_list = sorted(pivot_df.index.union(pivot_df.columns))
industry_to_index = {name: i for i, name in enumerate(industry_list)}
pivot_df.index = [industry_to_index[name] for name in pivot_df.index]
pivot_df.columns = [industry_to_index[name] for name in pivot_df.columns]

# === 控制 tick 显示：每隔 3 个 ===
n = 5
max_index = max(pivot_df.index.max(), pivot_df.columns.max())
ticks_to_show = list(range(0, max_index + 1, n))

# === 绘图 ===
plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    pivot_df,
    cmap="coolwarm",
    vmin=0,
    vmax=1,
    square=False,
    cbar_kws={"label": "Transferability Score", "shrink": 0.85, "aspect": 30}
)
ax.invert_yaxis()  # ✅ 翻转 y 轴

ax.set_xticks([i + 0.5 for i in ticks_to_show])
ax.set_yticks([i + 0.5 for i in ticks_to_show])
ax.set_xticklabels(ticks_to_show, rotation=45, ha="right", fontsize=14)
ax.set_yticklabels(ticks_to_show, fontsize=14)

plt.xlabel("Source Industry Index", fontsize=28)
plt.ylabel("Target Industry Index", fontsize=28)
plt.title("Transferability Score Heatmap Across Industries (indexed)", fontsize=28)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=16)
cbar.set_label("Transferability Score", fontsize=22)

# === 保存 ===
plot_path = os.path.join(output_folder, "transferability_heatmap_indexed_mean.png")
csv_matrix_path = os.path.join(output_folder, "transferability_matrix_indexed_mean.csv")
csv_indexmap_path = os.path.join(output_folder, "industry_index_mapping.csv")

plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()

# === 保存矩阵和 index 映射 ===
pivot_df.to_csv(csv_matrix_path)
pd.Series(industry_to_index).sort_values().to_csv(csv_indexmap_path, header=["Industry Index"])

print("✅ Heatmap saved to:", plot_path)
print("✅ Matrix CSV saved to:", csv_matrix_path)
print("✅ Industry index mapping saved to:", csv_indexmap_path)
