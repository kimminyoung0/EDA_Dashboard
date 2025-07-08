# eda_modules/cat_statistical_check.py

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm

def remove_outliers_zscore(df, col, threshold=7.0):
    z_scores = np.abs(stats.zscore(df[col].dropna()))
    return df[df[col].isin(df[col].dropna()[z_scores < threshold])]

def perform_multivariate_anova(df, cat_cols, num_col, z_threshold=7.0):
    """
    다변량 ANOVA (범주형 변수들 vs 수치형 변수) 분석 수행
    """
    df_clean = df.dropna(subset=[num_col] + cat_cols)
    df_clean = remove_outliers_zscore(df_clean, num_col, threshold=z_threshold)

    # 범주형 변수는 문자열로 변환
    for col in cat_cols:
        df_clean[col] = df_clean[col].astype(str)

    formula = f"{num_col} ~ " + " + ".join(cat_cols)
    try:
        model = ols(formula, data=df_clean).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        result_df = anova_table[['F', 'PR(>F)']].rename(columns={'F': 'F-statistic', 'PR(>F)': 'p-value'})
        result_df["Significant (<0.05)"] = result_df["p-value"] < 0.05
        return result_df.reset_index().rename(columns={'index': 'Variable'})
    except Exception as e:
        print(f"ANOVA 분석 실패: {e}")
        return None
