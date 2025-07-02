import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_value_distributions_by_item(df, cols, item_col="ITEM_CD", save_dir=None, color="skyblue"):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    saved_paths = []

    for item, group_df in df.groupby(item_col):
        item_dir = os.path.join(save_dir, str(item))
        os.makedirs(item_dir, exist_ok=True)

        for col in cols:
            img_path = os.path.join(item_dir, f"{col}_distribution_{color}.png")

            if os.path.exists(img_path):
                saved_paths.append(img_path)
                continue  # ✅ 이미지가 이미 있으면 재사용

            plt.figure(figsize=(8, 4))
            if pd.api.types.is_numeric_dtype(group_df[col]):
                if group_df[col].dropna().empty:
                    continue
                sns.histplot(group_df[col], kde=True, bins=30, color=color)
            else:
                value_counts = group_df[col].value_counts()
                if value_counts.empty:
                    continue
                value_counts.plot(kind='bar', color=color)

            plt.title(f"{item} - Distribution of {col}")
            plt.tight_layout()
            plt.savefig(img_path, facecolor='white')
            plt.close()

            saved_paths.append(img_path)

    return saved_paths
