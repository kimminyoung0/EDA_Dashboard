import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import hashlib
from io import StringIO
from typing import Dict, List

st.set_page_config(page_title="ITEM_CD별 변수 분포 비교", layout="centered")
st.title("📊 ITEM_CD별 컬럼 분포 비교 (여러 데이터)")

st.markdown("""
이 앱은 여러 개의 데이터를 업로드한 후, 특정 ITEM_CD와 변수(컬럼)를 선택하여  
각 데이터에서의 분포(KDE)를 하나의 그래프에 겹쳐 시각화할 수 있도록 합니다.
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

    # ✅ 전체 ITEM_CD (합집합)과 공통 수치형 컬럼 (교집합)
    all_itemcds = sorted(set().union(*[
        set(df["ITEM_CD"].dropna().unique())
        for df in data_dict.values() if "ITEM_CD" in df.columns
    ]))
    all_numerical_cols = set.intersection(*[
        set(df.select_dtypes(include='number').columns)
        for df in data_dict.values()
    ])

    if not all_itemcds:
        st.error("📛 모든 데이터에 ITEM_CD가 없습니다.")
    elif not all_numerical_cols:
        st.error("📛 모든 데이터에 공통된 수치형 변수가 없습니다.")
    else:
        selected_itemcd = st.selectbox("🔍 ITEM_CD 선택", all_itemcds)
        selected_col = st.selectbox("📈 비교할 수치형 변수 선택", sorted(all_numerical_cols))
        selected_color_palette = st.selectbox("🎨 색상 팔레트 선택", ["Set1", "Set2", "Dark2", "tab10"], index=0)
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

        # ✅ 캐시 키 생성: 데이터셋 이름 + 설정 정보 + x축 범위
        def generate_hash_key(file_names: List[str], item_cd: str, col: str, kde_option: bool, x_min, x_max) -> str:
            key_str = "_".join(sorted(file_names)) + f"_{item_cd}_{col}_{kde_option}"
            if set_xlim and x_min is not None and x_max is not None:
                key_str += f"_xlim_{x_min}_{x_max}"
            return hashlib.md5(key_str.encode()).hexdigest()

        hash_key = generate_hash_key(list(data_dict.keys()), selected_itemcd, selected_col, use_kde_for_constant, x_min, x_max)
        save_dir = f"reports_itemcd_col/distributions_by_item_compare/{selected_itemcd}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{selected_col}_compare_{hash_key}.png")

        if os.path.exists(save_path):
            st.image(save_path, caption=f"{selected_itemcd} - {selected_col} 저장된 분포 그래프", use_container_width=True)
        else:
            # ➕ 새로 시각화
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.set_palette(selected_color_palette)
            plotted = False

            for data_name, df in data_dict.items():
                if selected_col not in df.columns:
                    st.warning(f"⚠️ `{data_name}`: 선택한 컬럼 **`{selected_col}`** 이 존재하지 않습니다.")
                    continue

                sub_df = df[df["ITEM_CD"] == selected_itemcd]
                if sub_df.empty:
                    st.warning(f"⚠️ `{data_name}`: ITEM_CD **`{selected_itemcd}`** 에 해당하는 데이터가 없습니다.")
                    continue

                col_data = pd.to_numeric(sub_df[selected_col], errors='coerce').dropna()
                if len(col_data) == 0:
                    st.warning(f"⚠️ `{data_name}`: `{selected_col}` 컬럼에 유효한 데이터가 없습니다.")
                    continue

                unique_vals = col_data.unique()
                if len(unique_vals) == 1:
                    if use_kde_for_constant:
                        noise = np.random.normal(loc=0, scale=0.01, size=len(col_data))
                        noisy_data = col_data + noise
                        sns.kdeplot(noisy_data, label=data_name, ax=ax)
                    else:
                        sns.histplot(col_data, bins=1, label=data_name, ax=ax)
                elif len(col_data) > 1:
                    sns.kdeplot(col_data, label=data_name, ax=ax)

                plotted = True

            if plotted:
                ax.set_title(f"KDE distribution - ITEM_CD: {selected_itemcd}, Column: {selected_col}", fontsize=8)
                ax.set_xlabel(selected_col, fontsize=8)
                ax.set_ylabel("Density", fontsize=8)
                ax.legend(title="data", fontsize=8)
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
    st.info("⬅️ 좌측에서 CSV 파일을 업로드해주세요.")
