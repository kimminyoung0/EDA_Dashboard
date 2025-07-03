import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def plot_outliers_boxplot(df, col, save_path, color="skyblue"):
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col], color=color)
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, facecolor="white")
    plt.close()
    return save_path



def plot_outliers_zscore_custom(df, col, z_threshold, save_path, color="skyblue"):
    plt.figure(figsize=(6, 4))
    
    if df[col].nunique() == 1 or df[col].std() == 0:
        sns.countplot(x=df[col], color=color)
        plt.title(f"[{col}] 단일값 - 분산 0으로 이상치 판단 불가")
        plt.figtext(0.5, -0.1, "※ 이 컬럼은 단일값으로 Z-score 이상치 시각화가 불가능합니다.", 
                    ha="center", fontsize=9, color='red')
    else:
        mean = df[col].mean()
        std = df[col].std()
        lower = mean - z_threshold * std
        upper = mean + z_threshold * std

        sns.histplot(df[col], bins=30, kde=True, color=color)
        plt.axvline(lower, color='red', linestyle='--', label=f'-{z_threshold}σ')
        plt.axvline(upper, color='red', linestyle='--', label=f'+{z_threshold}σ')
        plt.title(f"Z-Score 이상치 시각화: {col}")
        plt.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, facecolor="white")
    plt.close()
    return save_path

def plot_outliers_iqr_custom(df, numerical_cols, save_path, q1: float = 25, q3: float = 75, k: float = 1.5, color="skyblue"):
    """
    IQR 커스텀 방식으로 이상치를 시각화
    """
    for col in numerical_cols:
        if col not in df.columns:
            print(f"없는 컬럼 {col}컬럼이 들어왔습니다.")
            continue
        plt.figure(figsize=(8, 4))
        q1_val = np.percentile(df[col], q1)
        q3_val = np.percentile(df[col], q3)
        iqr = q3_val - q1_val
        lower_bound = q1_val - k * iqr
        upper_bound = q3_val + k * iqr

        sns.histplot(df[col], bins=30, kde=True, color='lightgreen', label='Normal Range')
        plt.axvline(lower_bound, color='red', linestyle='--', label=f'Lower Bound ({round(lower_bound, 2)})')
        plt.axvline(upper_bound, color='red', linestyle='--', label=f'Upper Bound ({round(upper_bound, 2)})')
        plt.title(f"[{col}] IQR 이상치 (Q1={q1}, Q3={q3}, k={k})")
        plt.legend()
    plt.tight_layout()
    plt.close()
    return save_path
