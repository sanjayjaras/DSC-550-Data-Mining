"""
Script to find optimum k by elbow method and gap stats
"""

from sklearn import datasets
from sklearn import cluster as cl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snscccc


def calculate_sse(df, kmeans):
    """
    Calculate Sum squared deviation or erros
    :param df: input dataset
    :param kmeans: kmeans model
    :return: sum of squared errors
    """
    i = 0
    sse = 0
    while i < len(kmeans.cluster_centers_):
        mu = kmeans.cluster_centers_[i]
        ci = df[kmeans.labels_ == i]
        for xj in ci.iterrows():
            sse += np.linalg.norm(np.array(xj[1]) - mu) ** 2
        i += 1
    return sse


def calculate_distance_matrix(df):
    """
    calculate distance matrix for
    :param df: input dataset
    :return: distance matrix
    """
    n = df.shape[0]
    dist_matrix = np.zeros(shape=(150, 150))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(df[i] - df[j])
    return dist_matrix


def calculate_intra_cluster_weight(k, cis, dist_matrix, df):
    """
    calculate distance between all the points from each other in all clusters
    :param k: number of clusters
    :param cis: points in each cluster
    :param dist_matrix: distance matrix
    :param df: input dataset
    :return: intra-cluster weight sum in log 2 base 2 form
    """
    win = 0
    for ki in range(k):
        indices = df[cis == ki][0].index
        for i in indices:
            for j in indices:
                win += dist_matrix[i, j]
    return np.log2(win / 2)


def generate_random_n_samples(n, D):
    """
    Generate samples for calculating gap stats
    :param n: number of observation needed
    :param D: input dataset to find min and max for each feature
    :return: random sample Ri
    """
    mins = D.min()
    maxs = D.max()
    R = pd.DataFrame()
    i = 0
    for col in D.columns:
        R[col] = np.random.choice(np.random.uniform(mins[i], maxs[i], 300), n)
        i += 1
    return R


def calculate_win_k_r(df, k, t):
    """
    Calculate mu i.e. expected weight and deviation for k
    :param df: input data set this will random samples
    :param k: number of clusters
    :param t: number of times sampling will be done
    :return: expected intracluster weight and deviation
    """
    wks = []
    for i in range(t):
        r_sam = generate_random_n_samples(150, df)
        dist_mat_r = calculate_distance_matrix(np.array(r_sam))
        kmeans_r = cl.KMeans(n_clusters=k, init="random", max_iter=10)
        kmeans_r.fit(r_sam)
        win_k = calculate_intra_cluster_weight(k, kmeans_r.labels_, dist_mat_r, r_sam)
        wks.append(win_k)
    mu_wk = np.array(wks).sum() / t
    sigma_wk = ((((wks - mu_wk) ** 2).sum()) / t) ** 0.5
    return mu_wk, sigma_wk


def find_k_by_gap_stats(gap, k, sigmak):
    """
    find k from gap stats
    :param gap: gaps stats for all k's
    :param k: number of clusters
    :param sigmak: deviations in each cluster
    :return:
    """
    for i in range(k - 1):
        diff = gap[i + 1, 1] - sigmak[i + 1, 1]
        if gap[i, 1] >= diff:
            print(f"K found by Gap Stats is:{i + 1}")
            break
    print(f"Both stats(elbow plot and gap stats) shows clusters 3 & 4 when ran multiple times")


def main():
    """
    Main method
    :return:
    """
    np.random.seed(111)
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data[:])
    print(df.shape)
    t = 5
    k = 10
    dist_mat = calculate_distance_matrix(np.array(df))
    sses = np.empty((k, 2))
    wins = np.empty((k, 2))
    gap = np.empty((k, 2))
    mu_k = np.empty((k, 2))
    sigmak = np.empty((k, 2))
    for i in range(1, k + 1):
        print(f"Processing clusters:{i}")
        kmeans = cl.KMeans(n_clusters=i, init='random', max_iter=10)
        kmeans.fit(df)
        sse = calculate_sse(df, kmeans)
        sses[i - 1] = [i, sse]

        r_mu_wk, r_sigma_wk = calculate_win_k_r(df, i, t)
        mu_k[i - 1] = [i, r_mu_wk]
        sigmak[i - 1] = [i, r_sigma_wk]
        win_k = calculate_intra_cluster_weight(i, kmeans.labels_, dist_mat, df)
        gap_k = r_mu_wk - win_k
        gap[i - 1] = [i, gap_k]
        wins[i - 1] = [i, win_k]

    a4_dims = (11.7, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    ax = sns.lineplot(x=sses[:, 0], y=sses[:, 1], marker="X")
    ax.set(xticks=range(1, k + 1), title="SSE by clusters", ylabel="SSE", xlabel="Clusters")
    plt.show()

    ax = sns.lineplot(x=wins[:, 0], y=wins[:, 1], marker="X")
    ax = sns.lineplot(x=mu_k[:, 0], y=mu_k[:, 1], marker="*")
    ax.set(xticks=range(1, k + 1), title="Intracluster Weights", ylabel="gap(k)", xlabel="Clusters")
    plt.show()

    ax = sns.lineplot(x=gap[:, 0], y=gap[:, 1], marker="x")
    ax.set(xticks=range(1, k + 1), title="Gap Stats", ylabel="log2WinK", xlabel="Clusters")
    plt.show()

    find_k_by_gap_stats(gap, k, sigmak)


if __name__ == "__main__":
    """
    This will check if main method is called over the python file
    """
    main()
