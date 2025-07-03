import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_value_distributions(df, cols, item_col=None, save_dir=None, color="skyblue"):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    saved_paths = []

    groups = [(None, df)] if item_col is None else df.groupby(item_col)

    for item, group_df in groups:
        group_name = "all" if item is None else str(item)
        item_dir = os.path.join(save_dir, group_name)
        os.makedirs(item_dir, exist_ok=True)

        for col in cols:
            img_path = os.path.join(item_dir, f"{col}_distribution_{color}.png")

            if os.path.exists(img_path):
                saved_paths.append(img_path)
                continue

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

            plt.title(f"{group_name} - Distribution of {col}")
            plt.tight_layout()
            plt.savefig(img_path, facecolor='white')
            plt.close()

            saved_paths.append(img_path)

    return saved_paths
