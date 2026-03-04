 # eda_modules/scatter_plot.py

import pandas as pd
import numpy as np
import plotly.express as px  # type: ignore


def plot_scatter(df, x_col, y_col, hue_col=None, hover_all_cols: bool = True):
    """
    Plotly 산점도 (인터랙티브)

    Parameters:
        df: 데이터프레임
        x_col: X축 변수명
        y_col: Y축 변수명
        hue_col: 색상 구분 변수명 (선택사항)
        hover_all_cols: hover 시 모든 컬럼 정보를 보여줄지 여부

    Returns:
        Plotly Figure 또는 None
    """
    cols = [x_col, y_col] + ([hue_col] if hue_col else [])

    # x, y, hue에 결측이 없는 row만 남기되, 나머지 컬럼은 그대로 유지해서
    # hover에서 전체 row 정보를 볼 수 있게 함
    df_clean = df.dropna(subset=cols).copy()

    if df_clean.empty:
        return None, None

    # hover에 전체 row 정보를 넣고 싶으면 hover_data에 모든 컬럼 전달
    if hover_all_cols:
        hover_data = {col: True for col in df_clean.columns}
    else:
        hover_data = {col: True for col in cols}

    fig = px.scatter(
        df_clean,
        x=x_col,
        y=y_col,
        color=hue_col,
        hover_data=hover_data,
        opacity=0.7,
    )

    fig.update_layout(
        title=f"Scatter Plot: {x_col} vs {y_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        font=dict(family="Malgun Gothic"),
        legend_title=hue_col if hue_col else "",
        dragmode='select',  # 박스 선택 모드 활성화
        selectdirection='h',  # 수평/수직 선택 가능
    )

    fig.update_traces(
        marker=dict(size=7),
        selected=dict(marker=dict(size=10, color='red', opacity=1.0)),  # 선택된 점 강조
        unselected=dict(marker=dict(opacity=0.7))
    )

    return fig, df_clean  # df_clean도 반환해서 선택된 인덱스로 row를 찾을 수 있게 함
