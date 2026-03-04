# eda_modules/scatter_plot.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib  # type: ignore
import seaborn as sns
import os

# 한글 폰트 설정 (Windows 기준: Malgun Gothic)
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

def plot_scatter(df, x_col, y_col, hue_col=None, save_path=None, figsize=(8, 6)):
    """
    산점도 그리기
    
    Parameters:
        df: 데이터프레임
        x_col: X축 변수명
        y_col: Y축 변수명
        hue_col: 색상 구분 변수명 (선택사항)
        save_path: 저장 경로 (선택사항)
        figsize: 그래프 크기
    
    Returns:
        저장 경로 또는 None
    """
    df_clean = df[[x_col, y_col] + ([hue_col] if hue_col else [])].dropna()
    
    if len(df_clean) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if hue_col:
        sns.scatterplot(data=df_clean, x=x_col, y=y_col, hue=hue_col, ax=ax, alpha=0.6)
    else:
        sns.scatterplot(data=df_clean, x=x_col, y=y_col, ax=ax, alpha=0.6)
    
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    ax.set_title(f"Scatter Plot: {x_col} vs {y_col}", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return save_path
    else:
        return fig
