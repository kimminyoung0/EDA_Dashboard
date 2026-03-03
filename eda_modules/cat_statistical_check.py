# eda_modules/cat_statistical_check.py

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations

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

def perform_normality_test(df, num_col, cat_col=None, method='shapiro'):
    """
    정규성 검정 수행
    
    Parameters:
        df: 데이터프레임
        num_col: 수치형 변수명
        cat_col: 범주형 변수명 (None이면 전체 데이터에 대해 검정)
        method: 'shapiro' 또는 'ks' (Kolmogorov-Smirnov)
    
    Returns:
        결과 딕셔너리
    """
    results = {}
    
    if cat_col is None:
        # 전체 데이터에 대한 정규성 검정
        data = df[num_col].dropna()
        if len(data) < 3:
            return {"error": "데이터가 너무 적습니다 (최소 3개 필요)"}
        
        if method == 'shapiro':
            # Shapiro-Wilk 검정 (샘플 크기 제한: 5000)
            if len(data) > 5000:
                data = data.sample(5000, random_state=42)
            stat, p_value = stats.shapiro(data)
            test_name = "Shapiro-Wilk"
        else:  # ks
            # Kolmogorov-Smirnov 검정 (정규분포와 비교)
            stat, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
            test_name = "Kolmogorov-Smirnov"
        
        results = {
            "test": test_name,
            "group": "전체",
            "statistic": stat,
            "p_value": p_value,
            "is_normal": p_value > 0.05,
            "sample_size": len(data)
        }
    else:
        # 그룹별 정규성 검정
        groups = df[cat_col].dropna().unique()
        group_results = []
        
        for group in groups:
            group_data = df[df[cat_col] == group][num_col].dropna()
            if len(group_data) < 3:
                group_results.append({
                    "test": method,
                    "group": str(group),
                    "statistic": None,
                    "p_value": None,
                    "is_normal": None,
                    "sample_size": len(group_data),
                    "error": "데이터가 너무 적습니다"
                })
                continue
            
            if method == 'shapiro':
                if len(group_data) > 5000:
                    group_data = group_data.sample(5000, random_state=42)
                stat, p_value = stats.shapiro(group_data)
                test_name = "Shapiro-Wilk"
            else:  # ks
                stat, p_value = stats.kstest(group_data, 'norm', args=(group_data.mean(), group_data.std()))
                test_name = "Kolmogorov-Smirnov"
            
            group_results.append({
                "test": test_name,
                "group": str(group),
                "statistic": stat,
                "p_value": p_value,
                "is_normal": p_value > 0.05,
                "sample_size": len(group_data)
            })
        
        results = {"groups": group_results}
    
    return results

def perform_anova_with_posthoc(df, cat_col, num_col, z_threshold=7.0):
    """
    ANOVA 검정 및 사후검정 (Tukey HSD) 수행
    
    Parameters:
        df: 데이터프레임
        cat_col: 범주형 변수명 (단일 변수만)
        num_col: 수치형 변수명
        z_threshold: 이상치 제거 임계값
    
    Returns:
        (anova_result, posthoc_result) 튜플
    """
    df_clean = df.dropna(subset=[num_col, cat_col])
    df_clean = remove_outliers_zscore(df_clean, num_col, threshold=z_threshold)
    
    # 범주형 변수는 문자열로 변환
    df_clean[cat_col] = df_clean[cat_col].astype(str)
    
    # 그룹별 데이터 추출
    groups = df_clean[cat_col].unique()
    if len(groups) < 2:
        return None, {"error": "최소 2개 이상의 그룹이 필요합니다"}
    
    # ANOVA 수행
    try:
        formula = f"{num_col} ~ C({cat_col})"
        model = ols(formula, data=df_clean).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        # 인덱스에서 cat_col 찾기 (C()로 감싸져 있을 수 있음)
        idx_name = None
        for idx in anova_table.index:
            if cat_col in str(idx):
                idx_name = idx
                break
        
        if idx_name is None:
            # 첫 번째 행 사용 (보통 cat_col이 첫 번째)
            idx_name = anova_table.index[0] if len(anova_table) > 0 else None
        
        if idx_name is None:
            return None, {"error": "ANOVA 테이블에서 변수를 찾을 수 없습니다"}
        
        anova_result = {
            "F-statistic": anova_table.loc[idx_name, 'F'],
            "p-value": anova_table.loc[idx_name, 'PR(>F)'],
            "Significant (<0.05)": anova_table.loc[idx_name, 'PR(>F)'] < 0.05
        }
    except Exception as e:
        return None, {"error": f"ANOVA 분석 실패: {e}"}
    
    # 사후검정 (Tukey HSD)
    posthoc_result = None
    if anova_result["p-value"] < 0.05:  # 유의한 경우에만 사후검정
        try:
            tukey = pairwise_tukeyhsd(endog=df_clean[num_col], groups=df_clean[cat_col], alpha=0.05)
            posthoc_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            posthoc_result = {
                "summary": posthoc_df,
                "reject": tukey.reject
            }
        except Exception as e:
            posthoc_result = {"error": f"사후검정 실패: {e}"}
    
    return anova_result, posthoc_result

def perform_independent_ttest(df, cat_col, num_col, z_threshold=7.0):
    """
    독립표본 t검정 수행 (두 그룹 간 비교)
    
    Parameters:
        df: 데이터프레임
        cat_col: 범주형 변수명 (2개의 그룹만 있어야 함)
        num_col: 수치형 변수명
        z_threshold: 이상치 제거 임계값
    
    Returns:
        결과 딕셔너리
    """
    df_clean = df.dropna(subset=[num_col, cat_col])
    df_clean = remove_outliers_zscore(df_clean, num_col, threshold=z_threshold)
    
    # 범주형 변수는 문자열로 변환
    df_clean[cat_col] = df_clean[cat_col].astype(str)
    
    # 그룹별 데이터 추출
    groups = df_clean[cat_col].unique()
    if len(groups) != 2:
        return {"error": f"정확히 2개의 그룹이 필요합니다. 현재 그룹 수: {len(groups)}"}
    
    group1, group2 = groups[0], groups[1]
    data1 = df_clean[df_clean[cat_col] == group1][num_col].dropna()
    data2 = df_clean[df_clean[cat_col] == group2][num_col].dropna()
    
    if len(data1) < 2 or len(data2) < 2:
        return {"error": "각 그룹에 최소 2개 이상의 데이터가 필요합니다"}
    
    # 등분산성 검정 (Levene's test)
    levene_stat, levene_p = stats.levene(data1, data2)
    equal_var = levene_p > 0.05
    
    # t검정 수행
    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
    
    # 효과 크기 (Cohen's d)
    pooled_std = np.sqrt(((len(data1) - 1) * data1.std()**2 + (len(data2) - 1) * data2.std()**2) / 
                          (len(data1) + len(data2) - 2)))
    cohens_d = (data1.mean() - data2.mean()) / pooled_std if pooled_std > 0 else 0
    
    result = {
        "group1": group1,
        "group2": group2,
        "group1_mean": data1.mean(),
        "group2_mean": data2.mean(),
        "group1_std": data1.std(),
        "group2_std": data2.std(),
        "group1_n": len(data1),
        "group2_n": len(data2),
        "t-statistic": t_stat,
        "p-value": p_value,
        "Significant (<0.05)": p_value < 0.05,
        "equal_variance": equal_var,
        "levene_p": levene_p,
        "cohens_d": cohens_d
    }
    
    return result

def perform_ttest_posthoc(df, cat_col, num_col, z_threshold=7.0, alpha=0.05, correction='bonferroni'):
    """
    독립표본 t검정 사후검정 (여러 그룹 간 쌍별 비교)
    
    Parameters:
        df: 데이터프레임
        cat_col: 범주형 변수명
        num_col: 수치형 변수명
        z_threshold: 이상치 제거 임계값
        alpha: 유의수준
        correction: 다중비교 보정 방법 ('bonferroni', 'holm', 'fdr_bh', 'none')
    
    Returns:
        결과 데이터프레임
    """
    df_clean = df.dropna(subset=[num_col, cat_col])
    df_clean = remove_outliers_zscore(df_clean, num_col, threshold=z_threshold)
    
    # 범주형 변수는 문자열로 변환
    df_clean[cat_col] = df_clean[cat_col].astype(str)
    
    # 그룹별 데이터 추출
    groups = sorted(df_clean[cat_col].unique())
    if len(groups) < 2:
        return pd.DataFrame({"error": ["최소 2개 이상의 그룹이 필요합니다"]})
    
    # 모든 그룹 쌍에 대해 t검정 수행
    results = []
    for group1, group2 in combinations(groups, 2):
        data1 = df_clean[df_clean[cat_col] == group1][num_col].dropna()
        data2 = df_clean[df_clean[cat_col] == group2][num_col].dropna()
        
        if len(data1) < 2 or len(data2) < 2:
            continue
        
        # 등분산성 검정
        levene_stat, levene_p = stats.levene(data1, data2)
        equal_var = levene_p > 0.05
        
        # t검정
        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
        
        # 효과 크기
        pooled_std = np.sqrt(((len(data1) - 1) * data1.std()**2 + (len(data2) - 1) * data2.std()**2) / 
                              (len(data1) + len(data2) - 2)))
        cohens_d = (data1.mean() - data2.mean()) / pooled_std if pooled_std > 0 else 0
        
        results.append({
            "Group1": group1,
            "Group2": group2,
            "Group1_Mean": data1.mean(),
            "Group2_Mean": data2.mean(),
            "Group1_N": len(data1),
            "Group2_N": len(data2),
            "t-statistic": t_stat,
            "p-value": p_value,
            "Equal_Variance": equal_var,
            "Cohen's_d": cohens_d
        })
    
    if not results:
        return pd.DataFrame({"error": ["유효한 그룹 쌍이 없습니다"]})
    
    result_df = pd.DataFrame(results)
    
    # 다중비교 보정
    if correction != 'none':
        from statsmodels.stats.multitest import multipletests
        corrected = multipletests(result_df['p-value'], alpha=alpha, method=correction)
        result_df['p-value_corrected'] = corrected[1]
        result_df['Significant'] = corrected[0]
    else:
        result_df['p-value_corrected'] = result_df['p-value']
        result_df['Significant'] = result_df['p-value'] < alpha
    
    return result_df
