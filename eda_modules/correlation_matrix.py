# eda_modules/correlation_matrix.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_correlation_matrix(df, num_cols, save_path, method='pearson'):
    corr = df[num_cols].corr(method=method)

    plt.figure(figsize=(min(16, len(num_cols) * 1.2), len(num_cols) * 0.8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"label": f"{method.capitalize()} Correlation"})
    plt.title(f"{method.capitalize()} Correlation Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, facecolor="white")
    plt.close()

    return save_path
