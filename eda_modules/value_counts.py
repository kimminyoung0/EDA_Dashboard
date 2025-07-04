import pandas as pd
import streamlit as st

def show_value_counts(df: pd.DataFrame):
    st.subheader("📊 변수별 빈도 (value_counts)")

    with st.expander("🔍 변수 선택하여 빈도 확인"):
        selected_cols = st.multiselect("🎯 변수 선택 (1~4개)", options=df.columns.tolist(), max_selections=4)

    if not selected_cols:
        st.info("⬅️ 왼쪽에서 하나 이상의 변수를 선택해주세요.")
        return

    if len(selected_cols) > 4:
        st.warning("⚠️ 최대 4개 변수까지만 선택 가능합니다.")
        return

    # 그룹별 개수 계산
    group_counts = (
        df.groupby(selected_cols, dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(by=selected_cols)
        .reset_index(drop=True)
    )

    if len(selected_cols) >= 1:
        first_col = selected_cols[0]
        group_counts["__bg_color_group__"] = (
            group_counts[first_col] != group_counts[first_col].shift()
        ).cumsum()

        display_df = group_counts.drop(columns=["__bg_color_group__"])

        # 행별 색상 지정 함수
        def highlight_groups(row):
            color = "#f9f9f9" if group_counts.loc[row.name, "__bg_color_group__"] % 2 == 0 else "#ffffff"
            return ['background-color: {}; text-align: left'.format(color)] * len(row)

        styled_df = (
            display_df.style
            .apply(highlight_groups, axis=1)
            .set_table_styles(
                [
                    {
                        "selector": "th",
                        "props": [("text-align", "left"), ("white-space", "nowrap")]
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("text-align", "left"),
                            ("white-space", "pre-wrap"),
                            ("word-wrap", "break-word"),
                            ("min-width", "80px"),
                            ("max-width", "300px")
                        ]
                    },
                ]
            )
        )

        left, center, right = st.columns([1, 6, 1])
        with center:
            st.write("📋 선택 변수 기준 빈도표:")
            st.dataframe(styled_df, use_container_width=True, height=500)

    else:
        st.dataframe(group_counts, use_container_width=True)
