import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


def check_0_value(df, cols, cat_col, save_path):
    if cat_col not in df.columns:
        raise ValueError(f"{cat_col} is not in dataframe")

    zero_rate_df = (
        df[cols + [cat_col]]
        .groupby(cat_col)
        .apply(lambda x: (x[cols] == 0).mean())
    )

    plt.figure(figsize=(min(16, len(zero_rate_df.columns) * 1.2), len(zero_rate_df) * 0.5 + 1))
    sns.heatmap(zero_rate_df, annot=True, fmt=".2%", cmap="Blues", cbar_kws={"label": "0 value rate"}, vmin=0, vmax=1, annot_kws={"size": 5})
    plt.title("📊 0 Value Rate Heatmap")
    plt.ylabel(f"{cat_col} Column value")
    plt.xlabel(cat_col)
    plt.tight_layout()

    plt.savefig(save_path, facecolor="white")
    plt.close()

    return save_path

def check_null_value(df, cols, cat_col, save_path):
    if cat_col not in df.columns:
        raise ValueError(f"{cat_col} is not in dataframe")

    null_rate_df = (
        df[cols + [cat_col]]
        .groupby(cat_col)
        .apply(lambda x: x[cols].isnull().mean())
    )

    plt.figure(figsize=(min(16, len(null_rate_df.columns) * 1.2), len(null_rate_df) * 0.5 + 1))
    sns.heatmap(null_rate_df, annot=True, fmt=".2%", cmap="Reds", cbar_kws={"label": "Null rate"}, annot_kws={"size": 5})
    plt.title("📊 Null Value Rate Heatmap")
    plt.ylabel(f"{cat_col} Column value")
    plt.xlabel(cat_col)
    plt.tight_layout()
    plt.savefig(save_path, facecolor="white")
    plt.close()

    return save_path
