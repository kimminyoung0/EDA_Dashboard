import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def check_class_balance(df: pd.DataFrame, target_col: str):
    value_counts = df[target_col].value_counts(normalize=True).sort_index()
    print("🔍 클래스 비율 (%):")
    print((value_counts * 100).round(2))

    plt.figure(figsize=(6, 4))
    sns.barplot(x=value_counts.index.astype(str), y=value_counts.values)
    plt.title(f"Class Distribution of '{target_col}'")
    plt.xlabel("Class")
    plt.ylabel("Proportion")
    plt.ylim(0, 1)
    plt.show()
