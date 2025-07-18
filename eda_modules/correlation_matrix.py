# eda_modules/correlation_matrix.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_correlation_matrix(df, num_cols, save_path, method='pearson'):
    corr = df[num_cols].corr(method=method)

    n = len(num_cols) #엥겔 15 / 우진 25
    fig_size = max(16, n * 1.2) #max(8, n * 0.5)
    font_scale = 1.5 #min(1.0, 20 / n) 
    plt.figure(figsize=(fig_size, fig_size), dpi = 150)
    sns.set(font_scale=font_scale)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"label": f"{method.capitalize()} Correlation"})
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.title(f"{method.capitalize()} Correlation Matrix", fontsize=25, pad=70)
    plt.tight_layout()
    plt.savefig(save_path, facecolor="white")
    plt.close()

    return save_path
