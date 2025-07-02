import pandas as pd

def describe_by_type(df: pd.DataFrame, var_types: dict, save_path: str = None):
    """
    범주형, 수치형 변수 각각에 대해 describe() 통계 출력

    Parameters:
        df (pd.DataFrame): 전체 데이터프레임
        var_types (dict): {'categorical': [...], 'numerical': [...], 'datetime': [...]}
        save_path (str): 저장 경로 지정 시 CSV로 저장
    """
    results = {}

    # 1. 수치형 변수
    if var_types.get("numerical"):
        print("\n📊 수치형 변수 describe()")
        desc_num = df[var_types["numerical"]].describe()
        print(desc_num)
        results["numerical"] = desc_num
        if save_path:
            desc_num.to_csv(f"{save_path}/describe_numerical.csv")

    # 2. 범주형 변수
    if var_types.get("categorical"):
        print("\n🧩 범주형 변수 describe()")
        desc_cat = df[var_types["categorical"]].describe(include="all")
        print(desc_cat)
        results["categorical"] = desc_cat
        if save_path:
            desc_cat.to_csv(f"{save_path}/describe_categorical.csv")

    return results
