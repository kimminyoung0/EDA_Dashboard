import pandas as pd

def split_variable_types(df: pd.DataFrame, datetime_threshold=10):
    datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

    # 문자열 타입 컬럼만 날짜로 변환 시도
    for col in df.columns:
        if col not in datetime_cols and df[col].dtype == "object":
            try:
                parsed = pd.to_datetime(df[col], format="%Y-%m-%d", errors="raise")
            except Exception:
                parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.nunique() > datetime_threshold:
                datetime_cols.append(col)

    # 순서 유지하면서 분류
    categorical_cols = [col for col in df.columns 
                        if col not in datetime_cols and df[col].dtype in ["object", "category"]]
    
    numerical_cols = [col for col in df.columns 
                      if col not in datetime_cols and df[col].dtype not in ["object", "category", "datetime64[ns]"]]

    return {
        "datetime": datetime_cols,
        "categorical": categorical_cols,
        "numerical": numerical_cols
    }




