import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    df = pd.read_csv("caesarian.csv")
    dct = {}
    for col in df.columns:
        dct[col] = col.strip()
    df.rename(columns=dct, inplace=True)
    print(df.columns)

    print(df.caesarian.sum())

    sns.scatterplot(df.age, df.caesarian)
    plt.show()
    sns.scatterplot(df["delivery time"], df.caesarian)
    plt.show()

    print(df[df["blood pressure"] != 1]["caesarian"].value_counts())

    df.groupby(["caesarian"])
