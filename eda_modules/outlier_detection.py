import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

def plot_outliers_by_item(df, numerical_cols, item_col="ITEM_CD", save_dir=None, color="skyblue"):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    saved_paths = []

    for item, group_df in df.groupby(item_col):
        item_dir = os.path.join(save_dir, str(item))
        os.makedirs(item_dir, exist_ok=True)

        for col in numerical_cols:
            group_df[col] = pd.to_numeric(group_df[col], errors="coerce")
            img_path = os.path.join(item_dir, f"{col}_boxplot_{color}.png")

            if os.path.exists(img_path):
                saved_paths.append(img_path)
                continue

            plt.figure(figsize=(8, 4))
            sns.boxplot(x=group_df[col], color=color)

            # ✅ 일반 숫자 표기 적용 (강제 설정)
            ax = plt.gca()
            ax.ticklabel_format(style='plain', axis='x', useOffset=False)
            formatter = ScalarFormatter(useMathText=False)
            formatter.set_scientific(False)
            ax.xaxis.set_major_formatter(formatter)

            plt.title(f"Boxplot of {col} (ITEM_CD: {item})")
            plt.tight_layout()
            plt.savefig(img_path, facecolor='white')
            plt.close()

            saved_paths.append(img_path)

    return saved_paths

