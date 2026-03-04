import sys, os, json, re
import streamlit as st  # type: ignore
import pandas as pd
import numpy as np
import math
import matplotlib  # type: ignore

# 현재 파일 기준 상위 디렉토리(DataAnalysis)를 모듈 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eda_modules.variable_type_splitter import split_variable_types
from eda_modules.describe_by_type import describe_by_type
from eda_modules.correlation_matrix import plot_correlation_matrix
from eda_modules.outlier_detection import plot_outliers_boxplot, plot_outliers_iqr_custom, plot_outliers_zscore_custom
from eda_modules.class_balance_check import check_class_balance
from eda_modules.value_distribution import plot_value_distributions
from eda_modules.filters import filter_dataframe
from eda_modules.value_counts import show_value_counts
from eda_modules.categorical_heatmap import plot_cat_matrix
from eda_modules.null_0_value_check import check_0_value, check_null_value
from eda_modules.cat_statistical_check import (
    perform_multivariate_anova, 
    perform_normality_test,
    perform_anova_with_posthoc,
    perform_independent_ttest,
    perform_ttest_posthoc
)
from eda_modules.scatter_plot import plot_scatter
import streamlit.components.v1 as components  # type: ignore 

# Matplotlib 한글 폰트 설정 (Windows 기준: Malgun Gothic)
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# set_page_config: 앱의 초기 페이지 설정을 지정하는 함수
st.set_page_config(page_title="EDA Dashboard", layout="wide", #layout = centered
                    menu_items={
                        'Get Help': 'https://github.com/kimminyoung0',
                        'Report a bug': 'https://github.com/kimminyoung0/issues',
                        'About': 'EDA Dashboard by Kim Minyoung'
                    }) 

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

def sanitize_filename(name: str) -> str:
    # Windows에서 사용할 수 없는 문자를 ''로 대체
    return re.sub(r'[\\/*?:"<>|]', "", str(name))

# GA4 태그 삽입
components.html("""
<!-- Microsft Clarity -->
<script type="text/javascript">
    (function(c,l,a,r,i,t,y){
        c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
        t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
        y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
    })(window, document, "clarity", "script", "s8ns1np3w5");
</script>
""", height=0)



 
st.title("🧪 EDA 대시보드")

uploaded_file = st.file_uploader("📂 CSV 또는 Excel 파일 업로드", type=["csv", "xlsx", "xls"])

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
    
    # 엑셀 파일 읽기 (xls, xlsx 모두 지원)
    if ext == "csv":
        df = pd.read_csv(uploaded_file)
    elif ext in ["xlsx", "xls"]:
        # 엑셀 파일의 경우 엔진 자동 선택 (openpyxl 또는 xlrd)
        try:
            if ext == "xlsx":
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            else:  # xls
                df = pd.read_excel(uploaded_file, engine='xlrd')
        except Exception as e:
            try:
                # 대체 엔진 시도
                df = pd.read_excel(uploaded_file)
            except Exception as e2:
                st.error(f"❌ 엑셀 파일 읽기 실패: {e2}")
                st.stop()
    else:
        st.error(f"❌ 지원하지 않는 파일 형식입니다: {ext}")
        st.stop()
    
    df = sanitize_object_columns(df)
    st.success(f"✅ 파일 업로드 완료: {uploaded_file.name}")

    #st.set_page_config(page_title=data_name)
    st.title(f"📊 {data_name}")
    # data_title = "_".join(data_name.split("_")[3:])
    # st.set_page_config(page_title = data_title, layout="wide", #layout = centered
    #                 menu_items={
    #                     'Get Help': 'https://github.com/kimminyoung0',
    #                     'Report a bug': 'https://github.com/kimminyoung0/issues',
    #                     'About': 'EDA Dashboard by Kim Minyoung'
    #                 }) 

    report_base = os.path.join("reports", data_name)
    outlier_dir = os.path.join(report_base, "outliers_by_item")
    dist_dir = os.path.join(report_base, "distributions_by_item")

    os.makedirs(outlier_dir, exist_ok=True)
    os.makedirs(dist_dir, exist_ok=True)

    #제외할 컬럼 필터 불러오기
    filter_dir = "feature_filters"
    os.makedirs(filter_dir, exist_ok=True)
    FILTER_PATH = os.path.join(filter_dir, f"filter_config_{data_name}.json")
    stored_filters = load_filter_config(FILTER_PATH)

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

    # 탭으로 기능 분리
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 데이터 개요", 
        "📈 시각화", 
        "🧪 통계 검정", 
        "🔗 상관관계", 
        "📋 데이터 확인"
    ])

    with tab1:
        st.markdown("### 📊 데이터 개요")
        st.markdown("**기능:** 기술 통계, 변수 분류 확인, 변수별 개수 확인")
        
        st.subheader("📈 기술 통계 - Describe()")
        group_or_not = st.selectbox("🔍 그룹화 여부 선택", options=["그룹화 X", "그룹화"], key="selectbox_describe_group")
        if group_or_not == "그룹화":
            group_by_col = st.selectbox("🔍 그룹화할 컬럼 선택", options=filtered_var_types["categorical"], key="selectbox_describe_group_col")
            grouped_df = df.groupby(group_by_col)
            describe_result = describe_by_type(grouped_df, filtered_var_types)
        else:
            describe_result = describe_by_type(df, filtered_var_types)
        
        if "numerical" in describe_result:
            st.write("🔢 수치형 변수 통계")
            st.dataframe(describe_result["numerical"])
        if "categorical" in describe_result:
            st.write("🔤 범주형 변수 통계")
            st.dataframe(describe_result["categorical"])
        
        # 변수별 개수 확인 - 1개나 2개의 변수로 groupby해 개수 확인
        #num_cols = describe_result["numerical"].index.tolist() if isinstance(describe_result["numerical"], pd.DataFrame) else []
        #cat_cols = describe_result["categorical"].index.tolist() if isinstance(describe_result["categorical"], pd.DataFrame) else []
        show_value_counts(df)

    with tab2:
        st.markdown("### 📈 시각화")
        st.markdown("**기능:** 이상치 분포, 변수 분포(KDE), 범주형 히트맵, null/0값 확인, 산점도")
        
        # 이상치 분포 시각화
        st.subheader("🔍 이상치 분포 (boxplot) 시각화")
        outlier_method = st.selectbox("🧪 이상치 탐지 방식 선택", ["Boxplot(기본 IQR)", "Z-Score", "IQR"])
        with st.expander("🎨 이상치 boxplot 색상 설정"):
            selected_color_outlier = st.selectbox(
                "시각화 색상 선택",
                options = [
                    "skyblue", "orange", "green", "red", "purple", "black", "deepskyblue", 
                    "limegreen", "seagreen", "gray", "pink",
                    "gold", "navy", "coral", "dodgerblue", "darkorange", "darkred",
                    "olive", "teal", "mediumslateblue", "indigo", "crimson",
                    "chocolate", "darkgreen", "slategray", "orchid", "magenta",
                    "cadetblue", "lightseagreen", "firebrick", "mediumvioletred",
                    "tomato", "steelblue", "sienna", "peru", "turquoise"
                ],
                index=0,
                key="color_outlier"
            )

        # 범주형 변수 선택 (그룹화 기준)
        selected_groupby_col = st.selectbox("📑 이상치를 그룹화할 기준 범주형 변수 선택 (미선택 가능)", options=["선택 안함"] + categorical_cols, key="selectbox_outlier_group"
        )
        base_dir = "outlier_imgs"
        method_dir = {
            "Boxplot(기본 IQR)": "boxplot",
            "Z-Score": "zscore",
            "IQR": "iqr"
        }[outlier_method]

        if st.toggle("📦 이상치 시각화 보기", value=False, key=f"toggle_outlier_{selected_groupby_col}"):
            is_grouped = selected_groupby_col != "선택 안함"
            
            # 그룹 값 선택
            if is_grouped:
                group_values = df[selected_groupby_col].dropna().unique().tolist()
                selected_group_value = st.selectbox("🔍 확인할 값 선택", options=group_values, key="selectbox_outlier_value")
                df_selected = df[df[selected_groupby_col] == selected_group_value]
                save_dir = f"reports/{data_name}/outliers_by_{selected_groupby_col}/{method_dir}/"
                item_name = selected_group_value
            else:
                df_selected = df
                save_dir = f"reports/{data_name}/outliers_all/{method_dir}/"
                item_name = "all"

            # 시각화할 변수 선택
            selected_num_cols = st.multiselect(
                "🎯 시각화할 수치형 변수 선택",
                options=filtered_var_types["numerical"],
                default=filtered_var_types["numerical"],
                key="selectbox_outlier_columns"
            )

            if selected_num_cols:
                item_safe = re.sub(r'[\\/*?:"<>|]', "", str(item_name))
                item_dir = os.path.join(save_dir, item_safe)
                os.makedirs(item_dir, exist_ok=True)

                img_paths = []

                if outlier_method == "Boxplot(기본 IQR)":
                    param_dir = os.path.join(item_dir, f"basic_boxplot")
                    pass
                elif outlier_method == "Z-Score":
                    z_threshold = st.slider("Z-Score 임계값 (절댓값)", 1.0, 7.0, 3.0, step=1.0, key = f"z_threshold_slider")
                    param_dir = os.path.join(item_dir, f"zthreshold_{z_threshold}")
                elif outlier_method == "IQR":
                    q1 = st.slider("Q1 백분위 (하위 경계)", 25, 10, 25, step=5, key = f"q1_slider")
                    q3 = st.slider("Q3 백분위 (상위 경계)", 75, 90, 75, step=5, key = f"q3_slider")
                    k = st.slider("IQR 계수 (k)", 1.0, 3.0, 1.5, step=0.1, key = f"k_slider")
                    param_dir = os.path.join(item_dir, f"q1_{q1}_q3_{q3}_k_{k}")
                os.makedirs(param_dir, exist_ok=True)

                for col in selected_num_cols:
                    print("col", col)
                    img_path = os.path.join(param_dir, f"{col}_boxplot_{selected_color_outlier}.png")
                    if os.path.exists(img_path):
                        img_paths.append(img_path)
                    else:
                        # 새로 생성
                        if outlier_method == "Boxplot(기본 IQR)":
                            path = plot_outliers_boxplot(df_selected, col, save_path=img_path, color=selected_color_outlier)
                        elif outlier_method == "Z-Score":
                            path = plot_outliers_zscore_custom(df_selected, col, z_threshold=z_threshold, save_path=img_path, color=selected_color_outlier)
                        elif outlier_method == "IQR":
                            path = plot_outliers_iqr_custom(df_selected, col, save_path=img_path, q1=q1, q3=q3, k=k, color=selected_color_outlier)

                        img_paths.append(path)

                # 이미지 그리드 표시
                num_imgs = len(img_paths)
                n_cols = 1 if num_imgs == 1 else (2 if num_imgs == 2 else 3)
                for i in range(0, num_imgs, n_cols):
                    cols = st.columns(n_cols)
                    for j in range(n_cols):
                        if i + j < num_imgs:
                            with cols[j]:
                                st.image(img_paths[i + j], use_container_width=True)

        def sanitize_filename(name: str) -> str:
            return re.sub(r'[\\/*?:"<>|]', "", str(name))

        # KDE 그래프 시각화
        st.subheader("📊 각 변수 분포 kde 시각화")
        with st.expander("🎨 KDE 그래프 색상 설정"):
            selected_color_kde = st.selectbox(
                "색상 선택", 
                options = [
                    "skyblue", "orange", "green", "red", "purple", "black", "deepskyblue", 
                    "limegreen", "seagreen", "gray", "pink",
                    "gold", "navy", "coral", "dodgerblue", "darkorange", "darkred",
                    "olive", "teal", "mediumslateblue", "indigo", "crimson",
                    "chocolate", "darkgreen", "slategray", "orchid", "magenta",
                    "cadetblue", "lightseagreen", "firebrick", "mediumvioletred",
                    "tomato", "steelblue", "sienna", "peru", "turquoise"
                ],
                index=0, 
                key="color_kde"
            )

        # 🔸 그룹 기준 선택
        selected_groupby_col_kde = st.selectbox("📑 분포를 그룹화할 기준 범주형 변수 선택 (미선택 가능)", options=["선택 안함"] + categorical_cols, key="selectbox_kde_group"
        )

        if st.toggle("📦 각 변수 분포 kde 시각화 보기", value=False, key="toggle_kde_view"):
            kde_dir = f"reports/{data_name}/distributions_by_{selected_groupby_col_kde}"

            if selected_groupby_col_kde != "선택 안함": #ITEM_CD별로 시각화하고 싶을 때 
                group_values = df[selected_groupby_col_kde].dropna().unique().tolist()
                selected_value = st.selectbox(f"🔍 확인할 {selected_groupby_col_kde} 값 선택", options=group_values, key="selectbox_kde_value")

                df_selected = df[df[selected_groupby_col_kde] == selected_value]
                group_dir = os.path.join(kde_dir, f"{selected_groupby_col_kde}_{sanitize_filename(selected_value)}")
                os.makedirs(group_dir, exist_ok=True)
            else: #전체 데이터로 시각화하고 싶을 때 
                df_selected = df.copy()
                group_dir = os.path.join(kde_dir, "all")
                print("DEBUGGING: 전체 데이터로 시각화!!!!!!!")
                os.makedirs(group_dir, exist_ok=True)

            selected_cols_dist = st.multiselect(
                "🎯 시각화할 변수 선택",
                options=filtered_var_types["numerical"] + filtered_var_types["categorical"],
                default=filtered_var_types["numerical"] + filtered_var_types["categorical"],
                key="selectbox_kde_cols"
            )

            if selected_cols_dist:

                img_paths = []
                if (
                    os.path.exists(group_dir)
                    and any(fname.endswith(f"_distribution_{selected_color_kde}.png") for fname in os.listdir(group_dir))
                ):
                    for col in selected_cols_dist:
                        fname = f"{col}_distribution_{selected_color_kde}.png"
                        path = os.path.join(group_dir, fname)
                        if os.path.exists(path):
                            img_paths.append(path)
                else:
                    img_paths = plot_value_distributions(
                        df_selected,
                        selected_cols_dist,
                        item_col=None if selected_groupby_col_kde == "선택 안함" else selected_groupby_col_kde,
                        save_dir=kde_dir,
                        color=selected_color_kde
                    )

                # 시각화
                num_imgs = len(img_paths)
                n_cols = 1 if num_imgs == 1 else (2 if num_imgs == 2 else 3)
                for i in range(0, num_imgs, n_cols):
                    cols = st.columns(n_cols)
                    for j in range(n_cols):
                        if i + j < num_imgs:
                            with cols[j]:
                                st.image(img_paths[i + j], use_container_width=True)

        st.subheader("📊 범주형 변수 선택해서 Heatmap으로 데이터 분포 확인")
        selected_cat_cols = st.multiselect("🎯 확인할 변수 선택 (정확히 2개)", options=filtered_var_types["categorical"], key="selectbox_cat_heatmap")

        if len(selected_cat_cols) != 2:
            st.warning("⚠️ 정확히 두 개의 범주형 변수를 선택해주세요.")
        else:
            if st.toggle("📦 Heatmap 시각화 하기", value=False, key="toggle_heatmap"):
                cat_heatmap_dir = f"reports/{data_name}/categorical_heatmap/{selected_cat_cols[0]}_{selected_cat_cols[1]}"
                os.makedirs(cat_heatmap_dir, exist_ok=True)
                check_cols = filtered_var_types["numerical"]

                img_paths = []
                if (
                    os.path.exists(cat_heatmap_dir)
                    and any(fname.endswith("_median_heatmap.png") for fname in os.listdir(cat_heatmap_dir))
                ):
                    for col in check_cols:
                        fname = f"{col}_median_heatmap.png"
                        path = os.path.join(cat_heatmap_dir, fname)
                        if os.path.exists(path):
                            img_paths.append(path)
                else:
                    img_paths = plot_cat_matrix(df, selected_cat_cols, check_cols, save_path=cat_heatmap_dir)

                # 시각화
                num_imgs = len(img_paths)
                n_cols = 1 if num_imgs == 1 else 2
                for i in range(0, num_imgs, n_cols):
                    cols = st.columns(n_cols)
                    for j in range(n_cols):
                        if i + j < num_imgs:
                            with cols[j]:
                                st.image(img_paths[i + j])

                    st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("📊 null값 및 0값 확인하기")
        if st.toggle("📦 시작하기", value=False, key="toggle_null_zero"):
            selected_0_or_null = st.selectbox("🎯 0 / Null 확인 여부 선택", options=["0", "Null"], key="selectbox_0_or_null")
            selected_cat_idx_col = st.selectbox("🎯 확인할 변수 선택", options=filtered_var_types["categorical"], key="selectbox_cat_idx")
            null_0_dir = f"reports/{data_name}/null_0_value_check/"
            os.makedirs(null_0_dir, exist_ok=True)
            check_cols = filtered_var_types["numerical"]

            fname = f"{selected_0_or_null}_{selected_cat_idx_col}_value_check.png"
            img_path = os.path.join(null_0_dir, fname)

            # 이미지가 없을 경우에만 생성
            if not os.path.exists(img_path):
                if selected_0_or_null == "0":
                    img_path = check_0_value(df, check_cols, selected_cat_idx_col, save_path=img_path)
                elif selected_0_or_null == "Null":
                    img_path = check_null_value(df, check_cols, selected_cat_idx_col, save_path=img_path)

            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.warning("⚠️ 이미지 생성에 실패했습니다.")

    with tab3:
        st.markdown("### 🧪 통계 검정")
        st.markdown("**기능:** 정규성 검정, ANOVA 검정, 독립표본 t검정")
        
        st.subheader("📊 정규성 검정")

        if st.toggle("📦 정규성 검정 시작하기", value=False, key="toggle_normality"):
            selected_num_col_norm = st.selectbox(
                "🎯 검정할 수치형 변수 선택",
                options=filtered_var_types["numerical"],
                key="normality_num"
            )
            
            test_method = st.selectbox(
                "🧪 검정 방법 선택",
                options=["shapiro", "ks"],
                format_func=lambda x: "Shapiro-Wilk" if x == "shapiro" else "Kolmogorov-Smirnov",
                key="normality_method"
            )
            
            group_by_norm = st.selectbox(
                "📑 그룹별 검정 여부",
                options=["전체 데이터", "그룹별 검정"],
                key="normality_group"
            )
            
            selected_cat_col_norm = None
            if group_by_norm == "그룹별 검정":
                selected_cat_col_norm = st.selectbox(
                    "🎯 그룹화할 범주형 변수 선택",
                    options=filtered_var_types["categorical"],
                    key="normality_cat"
                )
            
            if st.button("🔍 정규성 검정 실행", key="btn_normality"):
                if selected_cat_col_norm:
                    result = perform_normality_test(df, selected_num_col_norm, cat_col=selected_cat_col_norm, method=test_method)
                else:
                    result = perform_normality_test(df, selected_num_col_norm, method=test_method)
                
                if "error" in result:
                    st.error(f"❌ {result['error']}")
                elif "groups" in result:
                    st.write("**그룹별 정규성 검정 결과**")
                    norm_df = pd.DataFrame(result["groups"])
                    st.dataframe(norm_df)
                else:
                    st.write("**정규성 검정 결과**")
                    st.json(result)

        st.subheader("📊 다변량 ANOVA (범주형 vs 수치형 변수)")

        with st.expander("ℹ️ ANOVA 검정 목적 및 사용 예시", expanded=False):
            st.markdown("""
        **ANOVA (분산분석) 검정의 목적:**
        - 여러 그룹(범주) 간 수치형 변수의 평균 차이가 통계적으로 유의한지 검정
        - 범주형 변수가 종속 변수(수치형)에 미치는 영향을 확인하여, 해당 범주형 변수가 타겟 변수를 예측하거나 설명하는 데 의미가 있는지 판단
        
        **사용 예시:**
        - 제품 카테고리별 판매량 차이 확인 → 카테고리가 판매량 예측에 유의한 변수인지 검증
        - 지역별 고객 만족도 점수 비교 → 지역이 만족도에 영향을 미치는지 확인
        - 교육 방법(A, B, C)별 시험 점수 비교 → 교육 방법이 성과에 차이를 만드는지 검증
        
        **일원배치 vs 이원배치:**
        - **일원배치 ANOVA**: 범주형 변수 1개 선택 시 (예: 지역별 판매량 비교)
        - **이원배치 ANOVA**: 범주형 변수 2개 선택 시 (예: 지역 × 제품유형별 판매량 비교, 상호작용 효과 확인)
        - **다원배치 ANOVA**: 범주형 변수 3개 이상 선택 시
            """)

        if st.toggle("📦 다변량 ANOVA 시작하기", value=False, key="toggle_multianova"):
            anova_mode = st.radio(
                "🔍 ANOVA 모드 선택",
                options=["다변량 ANOVA", "단일 변수 ANOVA + 사후검정"],
                key="anova_mode"
            )
            
            if anova_mode == "다변량 ANOVA":
                selected_cat_cols_anova = st.multiselect(
                    "🎯 독립 변수 (범주형) 선택", 
                    options=filtered_var_types["categorical"], 
                    key="multi_anova_cat"
                )

                selected_num_col_anova = st.selectbox(
                    "🎯 종속 변수 (수치형) 선택", 
                    options=filtered_var_types["numerical"], 
                    key="multi_anova_num"
                )

                # 일원배치/이원배치 안내
                if selected_cat_cols_anova:
                    num_factors = len(selected_cat_cols_anova)
                    if num_factors == 1:
                        st.info(f"ℹ️ 일원배치 ANOVA: 범주형 변수 1개 선택됨 ({selected_cat_cols_anova[0]})")
                    elif num_factors == 2:
                        st.info(f"ℹ️ 이원배치 ANOVA: 범주형 변수 2개 선택됨 ({', '.join(selected_cat_cols_anova)})")
                    else:
                        st.info(f"ℹ️ 다원배치 ANOVA: 범주형 변수 {num_factors}개 선택됨 ({', '.join(selected_cat_cols_anova)})")

                if selected_cat_cols_anova and selected_num_col_anova:
                    st.markdown(f"🧪 Z-Score 방식 이상치 제거 임계값: **7.0**")
                    result_df = perform_multivariate_anova(df, selected_cat_cols_anova, selected_num_col_anova, z_threshold=7.0)
                    if result_df is not None:
                        st.dataframe(result_df)
                    else:
                        st.warning("ANOVA 분석에 실패했거나 유효한 데이터가 없습니다.")
            else:  # 단일 변수 ANOVA + 사후검정
                selected_cat_col_anova = st.selectbox(
                    "🎯 독립 변수 (범주형, 단일 변수) 선택",
                    options=filtered_var_types["categorical"],
                    key="single_anova_cat"
                )
                
                selected_num_col_anova = st.selectbox(
                    "🎯 종속 변수 (수치형) 선택",
                    options=filtered_var_types["numerical"],
                    key="single_anova_num"
                )
                
                if selected_cat_col_anova and selected_num_col_anova:
                    st.markdown(f"🧪 Z-Score 방식 이상치 제거 임계값: **7.0**")
                    anova_result, posthoc_result = perform_anova_with_posthoc(
                        df, selected_cat_col_anova, selected_num_col_anova, z_threshold=7.0
                    )
                    
                    if anova_result:
                        st.write("**ANOVA 검정 결과**")
                        st.json(anova_result)
                        
                        if posthoc_result and "error" not in posthoc_result:
                            st.write("**사후검정 결과 (Tukey HSD)**")
                            st.dataframe(posthoc_result["summary"])
                        elif posthoc_result and "error" in posthoc_result:
                            st.warning(f"⚠️ {posthoc_result['error']}")
                        else:
                            st.info("ℹ️ ANOVA 결과가 유의하지 않아 사후검정을 수행하지 않았습니다 (p >= 0.05)")
                    else:
                        st.warning("ANOVA 분석에 실패했거나 유효한 데이터가 없습니다.")

        st.subheader("📊 독립표본 t검정")

        with st.expander("ℹ️ 독립표본 t검정 목적 및 사용 예시", expanded=False):
            st.markdown("""
        **독립표본 t검정의 목적:**
        - 두 개의 독립적인 그룹 간 수치형 변수의 평균 차이가 통계적으로 유의한지 검정
        - 범주형 변수로 나눈 두 집단의 평균을 비교하여, 해당 범주형 변수가 타겟 변수를 예측하거나 구분하는 데 유의한지 확인
        
        **사용 예시:**
        - 남성 vs 여성의 평균 소득 비교 → 성별이 소득에 영향을 미치는지 검증
        - A/B 테스트: 기존 버전 vs 새 버전의 전환율 비교 → 새 버전이 성과 개선에 효과가 있는지 확인
        - 치료군 vs 대조군의 회복 시간 비교 → 치료 효과가 있는지 검증
        - 고객 등급(일반/프리미엄)별 평균 구매금액 비교 → 등급이 구매 행동을 구분하는 의미 있는 변수인지 확인
            """)

        if st.toggle("📦 독립표본 t검정 시작하기", value=False, key="toggle_ttest"):
            ttest_mode = st.radio(
                "🔍 t검정 모드 선택",
                options=["두 그룹 비교", "여러 그룹 쌍별 비교 (사후검정)"],
                key="ttest_mode"
            )
            
            selected_cat_col_ttest = st.selectbox(
                "🎯 그룹 변수 (범주형) 선택",
                options=filtered_var_types["categorical"],
                key="ttest_cat"
            )
            
            selected_num_col_ttest = st.selectbox(
                "🎯 비교할 수치형 변수 선택",
                options=filtered_var_types["numerical"],
                key="ttest_num"
            )
            
            if ttest_mode == "두 그룹 비교":
                if selected_cat_col_ttest and selected_num_col_ttest:
                    st.markdown(f"🧪 Z-Score 방식 이상치 제거 임계값: **7.0**")
                    result = perform_independent_ttest(df, selected_cat_col_ttest, selected_num_col_ttest, z_threshold=7.0)
                    
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        st.write("**독립표본 t검정 결과**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(f"{result['group1']} 평균", f"{result['group1_mean']:.4f}")
                            st.metric(f"{result['group1']} 표준편차", f"{result['group1_std']:.4f}")
                            st.metric(f"{result['group1']} 샘플 수", result['group1_n'])
                        with col2:
                            st.metric(f"{result['group2']} 평균", f"{result['group2_mean']:.4f}")
                            st.metric(f"{result['group2']} 표준편차", f"{result['group2_std']:.4f}")
                            st.metric(f"{result['group2']} 샘플 수", result['group2_n'])
                        
                        st.write("**검정 통계량**")
                        st.json({
                            "t-statistic": result['t-statistic'],
                            "p-value": result['p-value'],
                            "Significant (<0.05)": result['Significant (<0.05)'],
                            "등분산성 (Levene's test p-value)": result['levene_p'],
                            "등분산 가정": result['equal_variance'],
                            "Cohen's d (효과 크기)": result['cohens_d']
                        })
            else:  # 여러 그룹 쌍별 비교
                correction_method = st.selectbox(
                    "🧪 다중비교 보정 방법",
                    options=["bonferroni", "holm", "fdr_bh", "none"],
                    format_func=lambda x: {
                        "bonferroni": "Bonferroni",
                        "holm": "Holm",
                        "fdr_bh": "FDR (Benjamini-Hochberg)",
                        "none": "보정 없음"
                    }[x],
                    key="ttest_correction"
                )
                
                if selected_cat_col_ttest and selected_num_col_ttest:
                    st.markdown(f"🧪 Z-Score 방식 이상치 제거 임계값: **7.0**")
                    result_df = perform_ttest_posthoc(
                        df, selected_cat_col_ttest, selected_num_col_ttest, 
                        z_threshold=7.0, correction=correction_method
                    )
                    
                    if "error" in result_df.columns:
                        st.error(f"❌ {result_df['error'].iloc[0]}")
                    else:
                        st.write("**쌍별 t검정 결과**")
                        st.dataframe(result_df)

    with tab4:
        st.markdown("### 🔗 상관관계 분석")
        st.markdown("**기능:** 수치형 변수 간 상관관계 분석, 산점도")
        
        st.subheader("📈 수치형 변수 간 상관관계 분석")

        if st.toggle("🔍 상관관계 분석 시작하기", key="toggle_corr"):
            corr_method = st.selectbox("📊 상관계수 계산 방식", options=["pearson", "spearman", "kendall"], key="corr_method")
            
            # 수치형 변수 선택
            selected_num_cols = st.multiselect("🎯 분석할 수치형 변수 선택", options=filtered_var_types["numerical"], default=filtered_var_types["numerical"])
            
            if selected_num_cols:
                corr_dir = f"reports/{data_name}/correlation_matrix/"
                os.makedirs(corr_dir, exist_ok=True)
                corr_img_path = os.path.join(corr_dir, f"{corr_method}_correlation_matrix.png")

                # 이미지 저장 및 출력
                plot_correlation_matrix(df, selected_num_cols, corr_img_path, method=corr_method)
                st.image(corr_img_path, use_container_width=True)

        st.subheader("📊 산점도 (Scatter Plot)")

        if st.toggle("📦 산점도 그리기", value=False, key="toggle_scatter"):
            selected_x_col = st.selectbox(
                "🎯 X축 변수 선택",
                options=filtered_var_types["numerical"],
                key="scatter_x"
            )
            
            selected_y_col = st.selectbox(
                "🎯 Y축 변수 선택",
                options=filtered_var_types["numerical"],
                key="scatter_y"
            )
            
            use_hue = st.checkbox("🎨 색상 구분 사용 (범주형 변수)", value=False, key="scatter_hue_check")
            
            selected_hue_col = None
            if use_hue:
                selected_hue_col = st.selectbox(
                    "🎯 색상 구분 변수 선택",
                    options=filtered_var_types["categorical"],
                    key="scatter_hue"
                )
            
            if selected_x_col and selected_y_col:
                # hover_all_cols는 기본값이 True이므로 별도 인자 없이 호출
                result = plot_scatter(df, selected_x_col, selected_y_col, selected_hue_col)
                
                # result가 튜플이고 첫 번째 요소가 None이 아닌지 확인
                fig_scatter = None
                df_clean = None
                
                if result is not None:
                    try:
                        # result가 튜플인지 확인
                        if hasattr(result, '__len__') and len(result) == 2:
                            if result[0] is not None:
                                fig_scatter, df_clean = result
                            else:
                                st.warning("⚠️ 유효한 데이터가 없어 산점도를 그릴 수 없습니다.")
                        else:
                            st.warning("⚠️ 산점도 함수가 예상과 다른 형식을 반환했습니다.")
                    except (TypeError, IndexError, AttributeError) as e:
                        st.warning(f"⚠️ 산점도 생성 중 오류가 발생했습니다: {e}")
                else:
                    st.warning("⚠️ 유효한 데이터가 없어 산점도를 그릴 수 없습니다.")
                
                if fig_scatter is not None and df_clean is not None:
                    
                    # 산점도 표시
                    event = st.plotly_chart(
                        fig_scatter, 
                        use_container_width=True, 
                        key="scatter_plot",
                        on_select="rerun"  # 선택 시 앱 재실행
                    )
                    
                    # 선택된 데이터 표시 영역
                    st.markdown("---")
                    st.subheader("📋 선택된 데이터")
                    st.caption("💡 그래프에서 점을 클릭하거나 드래그로 영역을 선택하면 해당 데이터가 아래에 표시됩니다.")
                    
                    # 선택 이벤트 처리
                    if event and "selection" in event:
                        try:
                            selection = event["selection"]
                            point_indices = []
                            
                            # Plotly의 선택 데이터에서 인덱스 추출
                            if "points" in selection:
                                for point in selection["points"]:
                                    # pointIndex 또는 pointNumber 사용
                                    idx = point.get("pointIndex") or point.get("pointNumber")
                                    if idx is not None:
                                        point_indices.append(int(idx))
                            
                            if point_indices:
                                # 중복 제거 및 정렬
                                point_indices = sorted(set(point_indices))
                                # 유효한 인덱스만 필터링
                                valid_indices = [i for i in point_indices if 0 <= i < len(df_clean)]
                                
                                if valid_indices:
                                    selected_rows = df_clean.iloc[valid_indices]
                                    st.dataframe(selected_rows, use_container_width=True)
                                    st.success(f"✅ {len(selected_rows)}개의 데이터 포인트가 선택되었습니다.")
                                else:
                                    st.info("📌 선택된 인덱스가 유효하지 않습니다.")
                            else:
                                st.info("📌 그래프에서 점을 클릭하거나 드래그로 영역을 선택해주세요.")
                        except Exception as e:
                            st.warning(f"⚠️ 데이터 선택 처리 중 오류: {e}")
                            st.info("📌 그래프에서 점을 다시 선택해주세요.")
                    else:
                        # 대안: 수동으로 인덱스 입력받기
                        st.markdown("**또는 수동으로 인덱스 입력:**")
                        selected_indices_input = st.text_input(
                            "선택할 행 인덱스 입력 (쉼표로 구분, 예: 0,5,10)",
                            key="manual_indices",
                            help="예: 0,5,10 또는 0-10 (범위)"
                        )
                        
                        if selected_indices_input:
                            try:
                                indices = []
                                # 범위 처리 (예: 0-10)
                                if '-' in selected_indices_input and ',' not in selected_indices_input:
                                    start, end = map(int, selected_indices_input.split('-'))
                                    indices = list(range(start, end + 1))
                                else:
                                    # 쉼표로 구분된 인덱스
                                    indices = [int(x.strip()) for x in selected_indices_input.split(',')]
                                
                                # 유효한 인덱스만 필터링
                                valid_indices = [i for i in indices if 0 <= i < len(df_clean)]
                                
                                if valid_indices:
                                    selected_rows = df_clean.iloc[valid_indices]
                                    st.dataframe(selected_rows, use_container_width=True)
                                    st.success(f"✅ {len(selected_rows)}개의 데이터 포인트가 표시되었습니다.")
                                else:
                                    st.warning("⚠️ 유효한 인덱스가 없습니다.")
                            except ValueError:
                                st.error("❌ 인덱스 형식이 올바르지 않습니다. 숫자만 입력해주세요.")
                            except Exception as e:
                                st.error(f"❌ 오류 발생: {e}")
                        else:
                            st.info("📌 그래프에서 점을 클릭하거나 드래그로 영역을 선택해주세요.")
                else:
                    st.warning("⚠️ 유효한 데이터가 없어 산점도를 그릴 수 없습니다.")

    with tab5:
        st.markdown("### 📋 데이터 확인")
        st.markdown("**기능:** 원본 데이터 확인, 필터링된 데이터 확인")
        
        st.subheader("📋 원본 데이터 확인")
        st.dataframe(df)

        # 필터링된 데이터 확인
        st.subheader("📋 필터링된 데이터 확인")
        filtered_df = filter_dataframe(df)
        st.dataframe(filtered_df)


else:
    st.info("⬆️ 분석을 시작하려면 파일을 업로드해주세요.")
