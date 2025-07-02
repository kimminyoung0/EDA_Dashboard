import sys, os, json
import streamlit as st
import pandas as pd
import numpy as np
import math

# 현재 파일 기준 상위 디렉토리(DataAnalysis)를 모듈 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eda_modules.variable_type_splitter import split_variable_types
from eda_modules.describe_by_type import describe_by_type
from eda_modules.correlation_matrix import plot_correlation_matrix
from eda_modules.outlier_detection import plot_outliers_by_item
from eda_modules.class_balance_check import check_class_balance
from eda_modules.value_distribution import plot_value_distributions_by_item
from eda_modules.filters import filter_dataframe
from eda_modules.value_counts import show_value_counts

# 제외할 컬럼 저장 및 불러오기(json)
def save_filter_config(filtered_vars, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(filtered_vars, f, ensure_ascii=False, indent=2)

def load_filter_config(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"datetime": [], "categorical": [], "numerical": []}

# Object 컬럼 streamlit에서 불필요한 충돌을 막기 위해 문자열로 모두 변경 처리
def sanitize_object_columns(df):
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).replace("nan", "")
    return df

# set_page_config: 앱의 초기 페이지 설정을 지정하는 함수
st.set_page_config(page_title="EDA Dashboard", layout="wide", #layout = centered
                    menu_items={
                        'Get Help': 'https://github.com/kimminyoung0',
                        'Report a bug': 'https://github.com/kimminyoung0/issues',
                        'About': 'EDA Dashboard by Kim Minyoung'
                    }) 
 
st.title("🧪 EDA 대시보드")

uploaded_file = st.file_uploader("📂 CSV 또는 Excel 파일 업로드", type=["csv", "xlsx"])

if uploaded_file:
    ext = uploaded_file.name.split(".")[-1].lower()
    data_name = uploaded_file.name.split(".")[0]
    data_path = os.path.join("data", uploaded_file.name)

    st.markdown(f"""
        <script>
            var newTitle = "{data_name} - eda 대시보드";
            document.title = newTitle;
        </script>
        """,
        unsafe_allow_html=True
    )

    os.makedirs("data", exist_ok=True)
    with open(data_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    df = pd.read_csv(uploaded_file) if ext == "csv" else pd.read_excel(uploaded_file)
    df = sanitize_object_columns(df)
    st.success(f"✅ 파일 업로드 완료: {uploaded_file.name}")

    st.title(f"📊 {data_name}")

    st.set_page_config(page_title = data_name, layout="wide", #layout = centered
                    menu_items={
                        'Get Help': 'https://github.com/kimminyoung0',
                        'Report a bug': 'https://github.com/kimminyoung0/issues',
                        'About': 'EDA Dashboard by Kim Minyoung'
                    }) 

    report_base = os.path.join("reports", data_name)
    outlier_dir = os.path.join(report_base, "outliers_by_item")
    dist_dir = os.path.join(report_base, "distributions_by_item")

    os.makedirs(outlier_dir, exist_ok=True)
    os.makedirs(dist_dir, exist_ok=True)

    #제외할 컬럼 필터 불러오기
    # FILTER_PATH = f"feature_filters/filter_config_{data_name}.json"
    # stored_filters = load_filter_config(FILTER_PATH)
    #제외할 컬럼 필터 불러오기
    Ffilter_dir = "feature_filters"
    os.makedirs(filter_dir, exist_ok=True)
    FILTER_PATH = os.path.join(filter_dir, f"filter_config_{data_name}.json")

    # 변수 유형 분류
    var_types = split_variable_types(df)
    datetime_cols = var_types["datetime"]
    categorical_cols = var_types["categorical"]
    numerical_cols = var_types["numerical"]

    st.markdown(f"### 데이터 날짜 정보")
    for col in datetime_cols:
        st.write(f"**{col}** 데이터 날짜 정보: **{df[col].min()}** ~ **{df[col].max()}**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📆 날짜형 컬럼")
        exclude_dt = st.multiselect("❌ 제외할 날짜형 컬럼", options=datetime_cols,
            default=stored_filters["datetime"], key="dt_ex")
        final_dt = [col for col in datetime_cols if col not in exclude_dt]
        for col in final_dt:
            st.write(f"- {col}")

    with col2:
        st.markdown("### 🔤 범주형 컬럼")
        exclude_cat = st.multiselect("❌ 제외할 범주형 컬럼", options=categorical_cols,
            default=stored_filters["categorical"], key="cat_ex")
        final_cat = [col for col in categorical_cols if col not in exclude_cat]
        for col in final_cat:
            st.write(f"- {col}")

    with col3:
        st.markdown("### 🔢 수치형 컬럼")
        exclude_num = st.multiselect( "❌ 제외할 수치형 컬럼", options=numerical_cols,
            default=stored_filters["numerical"], key="num_ex")
        final_num = [col for col in numerical_cols if col not in exclude_num]
        for col in final_num:
            st.write(f"- {col}")

    # 필터 저장 버튼
    if st.button("💾 현재 필터 저장"):
        save_filter_config({
            "datetime": exclude_dt,
            "categorical": exclude_cat,
            "numerical": exclude_num
        }, path = FILTER_PATH)
        st.success("✅ 필터 설정이 저장되었습니다.")

    filtered_var_types = {
        "datetime": final_dt,
        "categorical": final_cat,
        "numerical": final_num
    }

    st.subheader("📈 기술 통계 - Describe()")
    describe_result = describe_by_type(df, filtered_var_types)
    if "numerical" in describe_result:
        st.write("🔢 수치형 변수 통계")
        st.dataframe(describe_result["numerical"])
    if "categorical" in describe_result:
        st.write("🔤 범주형 변수 통계")
        st.dataframe(describe_result["categorical"])
    
    # 변수별 개수 확인 - 1개나 2개의 변수로 groupby해 개수 확인
    show_value_counts(df)

    with st.expander("🎨 이상치 boxplot 색상 설정"):
        selected_color_outlier = st.selectbox("시각화 색상 선택", options=["skyblue", "orange", "green", "red", "purple", "black", "deepskyblue", "limegreen", "seagreen", "gray", "pink"], index=0)

    # 이상치 분포 시각화
    st.subheader("📦 수치형 변수 이상치 분포 (ITEM_CD별)")
    if st.toggle("📦 이상치 시각화 보기", value=False):
        item_list = df["ITEM_CD"].dropna().unique().tolist()
        outlier_dir = f"reports/{data_name}/outliers_by_item"
        os.makedirs(outlier_dir, exist_ok=True)
        
        selected_item_outlier = st.selectbox("🔍 확인할 금형 선택", options=item_list, key="selectbox_outlier")
        st.markdown("""
            <style>
            .stMultiSelect > div {
                max-height: none;
                overflow-y: visible;
            }
            .stMultiSelect div[role="listbox"] {
                flex-wrap: wrap;
                max-height: 600px;
            }
            .stMultiSelect span {
                white-space: nowrap;
                overflow: visible;
                text-overflow: unset !important;
                max-width: none !important;
            }
            </style>
        """, unsafe_allow_html=True)

        selected_cols_outlier = st.multiselect("🎯 시각화할 수치형 변수 선택", options=filtered_var_types["numerical"], default=filtered_var_types["numerical"])
        if selected_item_outlier and selected_cols_outlier:
            df_selected = df[df["ITEM_CD"] == selected_item_outlier]
            item_dir = os.path.join(outlier_dir, str(selected_item_outlier))
            img_paths = []
            # 📌 1. 이미지가 이미 존재하면 재사용
            if (
                os.path.exists(item_dir)
                and any(fname.endswith(f"_boxplot_{selected_color_outlier}.png") for fname in os.listdir(item_dir))
            ):
                for col in selected_cols_outlier:
                    fname = f"{col}_boxplot_{selected_color_outlier}.png"
                    path = os.path.join(item_dir, fname)
                    if os.path.exists(path):
                        img_paths.append(path)
            else:
                # 📌 2. 이미지 없으면 새로 생성
                img_paths = plot_outliers_by_item(df_selected, selected_cols_outlier, item_col="ITEM_CD", save_dir=outlier_dir, color=selected_color_outlier)

            num_imgs = len(img_paths)
            n_cols = 1 if num_imgs == 1 else (2 if num_imgs == 2 else 3)

            for i in range(0, num_imgs, n_cols):
                cols = st.columns(n_cols)
                for j in range(n_cols):
                    if i + j < num_imgs:
                        with cols[j]:
                            st.image(img_paths[i + j], use_container_width=True)
    
    
    with st.expander("🎨 KDE 그래프 색상 설정"):
        selected_color_kde = st.selectbox(
            "색상 선택", 
            options=["skyblue", "orange", "green", "red", "purple", "black", "deepskyblue", "limegreen", "seagreen", "gray", "pink"], 
            index=0, 
            key="color_kde"
        )
    st.subheader("📊 각 변수 분포 시각화 (ITEM_CD별)")
    if st.toggle("📦 각 변수 분포 kde 시각화 보기", value=False):
        item_list = df["ITEM_CD"].dropna().unique().tolist()
        dist_dir = f"reports/{data_name}/distributions_by_item"
        os.makedirs(dist_dir, exist_ok=True)

        selected_item_dist = st.selectbox("🔍 확인할 금형 선택", options=item_list, key="selectbox_distribution") ###
        
        selected_cols_dist = st.multiselect("🎯 시각화할 변수 선택", options=filtered_var_types["numerical"] + filtered_var_types["categorical"], default=filtered_var_types["numerical"] + filtered_var_types["categorical"])
        
        if selected_item_dist and selected_cols_dist:
            df_selected = df[df["ITEM_CD"] == selected_item_dist]
            item_dir = os.path.join(dist_dir, str(selected_item_dist))
            img_paths = []

            if (
                os.path.exists(dist_dir)
                and any(fname.endswith(f"_distribution_{selected_color_kde}.png") for fname in os.listdir(dist_dir))
            ):
                for col in selected_cols_dist:
                    fname = f"{col}_distribution_{selected_color_kde}.png"
                    path = os.path.join(item_dir, fname)
                    if os.path.exists(path):
                        img_paths.append(path)
            else:
                # 📌 2. 이미지 없으면 새로 생성
                img_paths = plot_value_distributions_by_item(df_selected, selected_cols_dist, item_col="ITEM_CD", save_dir=dist_dir, color=selected_color_kde)

            num_imgs = len(img_paths)
            n_cols = 1 if num_imgs == 1 else (2 if num_imgs == 2 else 3)

            for i in range(0, num_imgs, n_cols):
                cols = st.columns(n_cols)
                for j in range(n_cols):
                    if i + j < num_imgs:
                        with cols[j]:
                            st.image(img_paths[i + j], use_container_width=True)
    

    st.subheader("📊 상관관계 분석 (금형별)")

    st.subheader("📋 원본 데이터 확인")
    st.dataframe(df)

    # 필터링된 데이터 확인
    st.subheader("📋 필터링된 데이터 확인")
    filtered_df = filter_dataframe(df)
    st.dataframe(filtered_df)

else:
    st.warning("⬆️ 분석을 시작하려면 먼저 파일을 업로드해주세요.")
