import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use("seaborn-darkgrid")
plt.rcParams["figure.figsize"] = [24, 12]


def fraction_of_volume_sphere(center=0, l=2):
    """
    Find fraction of points lies in inscribed hypersphere with radius 1 inside the hypercube with edge length 2
    :param center:
    :param l:
    :return:
    """
    num_of_rand_points = 10000
    res = pd.DataFrame(columns=["d", "fraction"])
    for i in range(1, 101):
        df = pd.DataFrame()
        df["Center"] = pd.Series([center for _ in range(num_of_rand_points)])
        for j in range(1, i + 1):
            arr = np.random.uniform(-(l / 2), l / 2, num_of_rand_points)
            col = pd.Series(arr)
            df['X' + str(j)] = col
        dist_df = pd.DataFrame()
        for col in df.columns:
            if col is not "Center":
                dist_df["Diff_" + col] = ((df[col] - df["Center"]) ** 2)
        sum = dist_df.sum(axis=1)
        dist = sum ** (1 / 2)
        inside_sp = dist <= 1
        fraction = inside_sp.sum() / num_of_rand_points
        res = res.append(pd.DataFrame({"d": i, "fraction": [fraction]}))
        # print(f"d:{i}\tfraction:{fraction}")
    plot_fraction_of_volume_sphere(res)


def plot_fraction_of_volume_sphere(df):
    """
    Plot the graph that shows fraction points in hypersphere by dimension
    :param df:
    :return:
    """
    plt.plot(df.d, df.fraction, label="fraction")
    df["col"] = np.where(df.fraction <= 0, 'r', 'b')
    plt.scatter(df.d, df.fraction, c=df.col, edgecolor='none')
    plt.xticks(range(1, 100, 5))
    plt.title("Fraction of points in hypersphere")
    plt.xlabel("d")
    plt.ylabel("fraction")
    plt.legend()
    plt.show()


def fraction_of_volume_shell(center=0, l=2, eps=0.01):
    """
    Find number of points those are inside shell with eps(default:0.01)
    :param center:
    :param l:
    :param eps:
    :return:
    """
    num_of_rand_points = 10000
    res = pd.DataFrame(columns=["d", "fraction"])
    for i in range(1, 2000, 100):
        df = pd.DataFrame()
        df["Center"] = pd.Series([center for _ in range(num_of_rand_points)])
        for j in range(1, i + 1):
            arr = np.random.uniform(-(l / 2), l / 2, num_of_rand_points)
            col = pd.Series(arr)
            df['X' + str(j)] = col
        abs_col = df.abs()
        max = abs_col.max(axis=1)
        inside_shell = np.where((max >= (1 - eps)) & (max <= 1), True, False)
        fraction = inside_shell.sum() / num_of_rand_points
        res = res.append(pd.DataFrame({"d": [i], "fraction": [fraction]}))
        # print(f"d:{i}\tfraction:{fraction}")
    plot_fraction_of_volume(res)


def plot_fraction_of_volume(df):
    """
    Plot the graph showing fraction points inside the hypercube shell
    :param df:
    :return:
    """
    plt.plot(df.d, df.fraction, label="fraction")
    df["col"] = np.where(df.fraction >= 0.9999, 'r', 'b')
    plt.scatter(df.d, df.fraction, c=df.col, edgecolor='none')
    plt.xticks(range(1, 2000, 100))
    plt.title("Fraction of Volume inside hypercube shell")
    plt.xlabel("d")
    plt.ylabel("fraction")
    plt.legend()
    plt.show()


def main():
    """
    Maim method that will call other methods
    :return:
    """
    print("The dimensions for which volume in the sphere goes to zero are shown in Red Dots")
    fraction_of_volume_sphere()
    print("The dimensions for which volume in the shell goes to 100% are shown in Red Dots")
    fraction_of_volume_shell()
    

# Check for main is called on this file
if __name__ == "__main__":
    ''' This will check if main method is called over the python file
    '''
    print("Processing...")
    # call main function
    main()
