import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import hashlib
from io import StringIO
from typing import Dict, List

import matplotlib.colors as mcolors

CUSTOM_PALETTES = {
    "tab10": sns.color_palette("tab10", 10),
    "tab20": sns.color_palette("tab20", 20),
    "Set1": sns.color_palette("Set1", 9),
    "Set2": sns.color_palette("Set2", 8),
    "Dark2": sns.color_palette("Dark2", 8),
    "colorblind": sns.color_palette("colorblind", 10),
    "custom20": list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())[:10]
}

st.set_page_config(page_title="여러 데이터 분포 비교", layout="wide")
st.title("📊 여러 데이터 분포 비교를 위한 대시보드")

st.markdown("""
이 앱은 여러 개의 데이터를 업로드한 후, 원하는 **범주형 변수 기준으로 수치형 변수의 분포(KDE)** 를 시각화할 수 있도록 합니다.
""")

# 📂 데이터 업로드
uploaded_files = st.file_uploader("📂 여러 데이터 파일 업로드 (CSV)", type="csv", accept_multiple_files=True)
data_dict: Dict[str, pd.DataFrame] = {}

if uploaded_files:
    for file in uploaded_files:
        file_name = os.path.splitext(file.name)[0]
        stringio = StringIO(file.getvalue().decode("utf-8"))
        df = pd.read_csv(stringio)
        data_dict[file_name] = df

    # ✅ 공통 범주형 / 수치형 변수 확인
    common_cat_cols = set.intersection(*[
        set(df.select_dtypes(include='object').columns)
        for df in data_dict.values()
    ])
    common_num_cols = set.intersection(*[
        set(df.select_dtypes(include='number').columns)
        for df in data_dict.values()
    ])

    if not common_cat_cols:
        st.error("📛 공통된 범주형 변수가 없습니다.")
    elif not common_num_cols:
        st.error("📛 공통된 수치형 변수가 없습니다.")
    else:
        selected_cat_col = st.selectbox("🧩 기준이 될 범주형 변수 선택", sorted(common_cat_cols))
        selected_col = st.selectbox("📈 비교할 수치형 변수 선택", sorted(common_num_cols))
        selected_color_palette = st.selectbox("🎨 색상 팔레트 선택", list(CUSTOM_PALETTES.keys()), index=0)
        use_kde_for_constant = st.checkbox("📌 단일값 컬럼에 KDE 그리기 (노이즈 포함)", value=False)

        # ✅ X축 범위 직접 설정
        set_xlim = st.toggle("🧭 X축 범위 직접 지정하기", value=False)
        x_min, x_max = None, None
        if set_xlim:
            col1, col2 = st.columns(2)
            with col1:
                x_min = st.number_input("X축 최소값", value=0.0)
            with col2:
                x_max = st.number_input("X축 최대값", value=300.0)

        # ✅ 캐시 키 생성
        def generate_hash_key(file_names: List[str], cat_col: str, col: str, kde_option: bool, x_min, x_max) -> str:
            key_str = "_".join(sorted(file_names)) + f"_{cat_col}_{col}_{kde_option}"
            if set_xlim and x_min is not None and x_max is not None:
                key_str += f"_xlim_{x_min}_{x_max}"
            return hashlib.md5(key_str.encode()).hexdigest()

        hash_key = generate_hash_key(list(data_dict.keys()), selected_cat_col, selected_col, use_kde_for_constant, x_min, x_max)
        save_dir = f"reports_catcol/distributions_by_category_compare/{selected_cat_col}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{selected_col}_compare_{hash_key}.png")

        if os.path.exists(save_path):
            st.image(save_path, caption=f"{selected_cat_col} - {selected_col} 저장된 분포 그래프", use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.set_palette(CUSTOM_PALETTES[selected_color_palette])
            plotted = False

            for data_name, df in data_dict.items():
                if selected_cat_col not in df.columns or selected_col not in df.columns:
                    continue

                for group_val, group_df in df.groupby(selected_cat_col):
                    col_data = pd.to_numeric(group_df[selected_col], errors='coerce').dropna()
                    if len(col_data) == 0:
                        continue

                    label = f"{data_name} - {group_val}"
                    unique_vals = col_data.unique()

                    if len(unique_vals) == 1:
                        if use_kde_for_constant:
                            noise = np.random.normal(loc=0, scale=0.01, size=len(col_data))
                            noisy_data = col_data + noise
                            sns.kdeplot(noisy_data, label=label, ax=ax)
                        else:
                            sns.histplot(col_data, bins=1, label=label, ax=ax)
                    elif len(col_data) > 1:
                        sns.kdeplot(col_data, label=label, ax=ax)

                    plotted = True

            if plotted:
                ax.set_title(f"KDE 분포: {selected_cat_col}별 {selected_col}", fontsize=10)
                ax.set_xlabel(selected_col, fontsize=10)
                ax.set_ylabel("Density", fontsize=10)
                ax.legend(title="Data - Group", fontsize=8)
                ax.grid(True, linestyle="--", alpha=0.4)

                if set_xlim and x_min is not None and x_max is not None and x_min < x_max:
                    ax.set_xlim(x_min, x_max)

                plt.tight_layout()
                st.pyplot(fig)

                fig.savefig(save_path)
                st.success(f"✅ 그래프가 저장되었습니다: {save_path}")
            else:
                st.warning("⚠️ 모든 데이터셋에서 시각화 가능한 데이터가 없어 그래프를 생성하지 못했습니다.")
else:
    st.info("⬆️ 분석을 시작하려면 파일을 업로드해주세요.")
