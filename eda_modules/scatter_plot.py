# eda_modules/scatter_plot.py

import pandas as pd
import numpy as np
import plotly.express as px  # type: ignore

def plot_scatter(df, x_col, y_col, hue_col=None):
    """
    Plotly 산점도 (인터랙티브)
    
    Parameters:
        df: 데이터프레임
        x_col: X축 변수명
        y_col: Y축 변수명
        hue_col: 색상 구분 변수명 (선택사항)
    
    Returns:
        tuple: (fig, df_clean) 튜플을 반환
               - fig: Plotly Figure 객체 (데이터가 없으면 None)
               - df_clean: 정제된 데이터프레임 (데이터가 없으면 None)
    """
    # 입력 검증
    if df is None or df.empty:
        return None, None
    
    if x_col not in df.columns or y_col not in df.columns:
        return None, None
    
    if hue_col is not None and hue_col not in df.columns:
        return None, None
    
    cols = [x_col, y_col] + ([hue_col] if hue_col else [])
    
    # x, y, hue에 결측이 없는 row만 남기되, 나머지 컬럼은 그대로 유지
    df_clean = df.dropna(subset=cols).copy()
    
    if df_clean.empty:
        return None, None
    
    # Plotly 산점도 생성
    fig = px.scatter(
        df_clean,
        x=x_col,
        y=y_col,
        color=hue_col,
        hover_data={x_col: True, y_col: True},  # hover에 x, y 값 표시
        opacity=0.7,
    )
    
    # 레이아웃 설정
    fig.update_layout(
        title=f"Scatter Plot: {x_col} vs {y_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        font=dict(family="Malgun Gothic"),
        legend_title=hue_col if hue_col else "",
    )
    
    # 마커 크기 설정
    fig.update_traces(marker=dict(size=7))
    
    return fig, df_clean
