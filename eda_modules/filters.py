import pandas as pd
import streamlit as st

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Streamlit에서 변수별 필터링 UI 제공 및 필터 적용"""
    st.subheader("📋 원본 데이터 미리보기 및 필터링")

    with st.expander("🔎 변수별 필터링"):
        filter_cols = st.multiselect("필터링할 컬럼 선택", df.columns.tolist())

        filtered_df = df.copy()

        for col in filter_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val, max_val = df[col].min(), df[col].max()
                selected_range = st.slider(f"{col} 범위 선택", float(min_val), float(max_val), (float(min_val), float(max_val)))
                filtered_df = filtered_df[filtered_df[col].between(*selected_range)]

            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == object:
                unique_vals = df[col].dropna().unique().tolist()
                selected_vals = st.multiselect(f"{col} 값 선택", options=unique_vals, default=unique_vals)
                filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                min_date, max_date = df[col].min(), df[col].max()
                selected_date = st.date_input(f"{col} 날짜 선택", (min_date, max_date))
                if isinstance(selected_date, tuple) and len(selected_date) == 2:
                    filtered_df = filtered_df[df[col].between(selected_date[0], selected_date[1])]

    return filtered_df
