import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use("seaborn-darkgrid")
plt.rcParams["figure.figsize"] = [24, 12]


def nearest_neighbor(center=0.5):
    """
    generate random points in unit hypercube with dimension 1-100 and edge and find distance for each point from center
    :param center:
    :return:
    """
    num_of_rand_points = 10000
    res = pd.DataFrame(columns=["dn", "df", "ratio"])
    for i in range(1, 101):
        df = pd.DataFrame()
        df["Center"] = pd.Series([center for _ in range(num_of_rand_points)])
        for j in range(1, i + 1):
            arr = np.random.uniform(0, 1, num_of_rand_points)
            col = pd.Series(arr)
            df['X' + str(j)] = col
        dist_df = pd.DataFrame()
        for col in df.columns:
            if col is not "Center":
                dist_df["Diff_" + col] = ((df[col] - df["Center"]) ** 2)
        sum = dist_df.sum(axis=1)
        dist = sum ** (1 / 2)
        dn = min(dist)
        df = max(dist)
        ratio = dn / df
        res = res.append(pd.DataFrame({"dn": [dn], "df": [df], "ratio": [ratio]}), ignore_index=True)
    plot_nearest_neighbor(res)


def plot_nearest_neighbor(df):
    """
    Plot the graph for nearest points, furthest points, and ratio between dn/df
    :param df:
    :return:
    """
    plt.plot(range(1, len(df) + 1), df.dn, marker="o", label="dn")
    plt.plot(range(1, len(df) + 1), df.df, marker="o", label="df")
    plt.plot(range(1, len(df) + 1), df.ratio, marker="o", label="ratio dn/df")
    plt.xticks(range(0, len(df) + 1, 5))
    plt.title("Nearest Neighbor distance")
    plt.xlabel("d")
    plt.ylabel("distance")
    plt.legend()
    plt.show()


def main():
    """
    Maim method that will call other methods
    :return:
    """
    nearest_neighbor()


# Check for main is called on this file
if __name__ == "__main__":
    ''' This will check if main method is called over the python file
    '''
    print("Processing...")
    # call main function
    main()
