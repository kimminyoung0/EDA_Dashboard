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
    num_cats = len(zero_rate_df)       # y축 (범주 개수)
    num_vars = len(zero_rate_df.columns)
    fig_width = min(0.6 * num_vars, 30)   # 변수 개수에 비례하되 최대 30
    fig_height = min(0.6 * num_cats, 50)  # 범주 개수에 비례하되 최대 20

    # plt.figure(figsize=(min(16, len(zero_rate_df.columns) * 1.2), len(zero_rate_df) * 0.5 + 1))
    plt.figure(figsize=(fig_width, fig_height), dpi = 150)
    #sns.set(font_scale=2.0)
    sns.heatmap(zero_rate_df, annot=True, fmt=".1%", cmap="Blues", cbar_kws={"label": "0 value rate"}, vmin=0, vmax=1, annot_kws={"size": 7}) #
    plt.title("0 Value Rate Heatmap", fontsize=15, pad=30)
    plt.xticks(fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=10)
    plt.ylabel(f"{cat_col} Column value", fontsize=10)
    plt.xlabel(cat_col, fontsize=10)
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

    num_cats = len(null_rate_df)       # y축 (범주 개수)
    num_vars = len(null_rate_df.columns)
    fig_width = min(0.6 * num_vars, 30)   # 변수 개수에 비례하되 최대 30
    fig_height = min(0.6 * num_cats, 30)

    plt.figure(figsize=(fig_width, fig_height), dpi = 150)
    sns.heatmap(null_rate_df, annot=True, fmt=".1%", cmap="Reds", cbar_kws={"label": "Null rate"}, vmin=0, vmax=1, annot_kws={"size": 7})
    plt.title("Null Value Rate Heatmap", fontsize=15, pad=30)
    plt.xticks(fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=10)
    plt.ylabel(f"{cat_col} Column value", fontsize=10)
    plt.xlabel(cat_col, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, facecolor="white")
    plt.close()

    return save_path
