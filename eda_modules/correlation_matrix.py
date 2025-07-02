import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(df: pd.DataFrame, numerical_cols: list, save_path: str = None):
    corr = df[numerical_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix of Numerical Features")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        pass
        #plt.show()
