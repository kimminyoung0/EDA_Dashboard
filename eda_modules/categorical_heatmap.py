import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


def plot_cat_matrix(df, cat_cols, num_cols, save_path):
    img_paths = []

    for col in num_cols:
        pivot_df = df.pivot_table(
            index=cat_cols[0],
            columns=cat_cols[1],
            values=col,
            aggfunc='median'
        )
        # if mfr == "woojin":
        #     param = 0.8
        # elif mfr == "engel":
        #     param = 0.4

        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap='coolwarm', cbar_kws={'label': f'{col} 중앙값'})
        plt.title(f'{cat_cols[0]} : {cat_cols[1]} - {col} Median Heatmap')
        plt.xlabel(cat_cols[1])
        plt.ylabel(cat_cols[0])
        plt.tight_layout()

        filename = f"{col}_median_heatmap.png"
        img_path = os.path.join(save_path, filename)
        plt.savefig(img_path, facecolor="white")
        plt.close()

        img_paths.append(img_path)

    return img_paths
