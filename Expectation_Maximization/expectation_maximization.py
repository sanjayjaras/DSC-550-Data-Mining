"""
This program is to implement Expectation Maximization(EM) algorithm from the book
Data Mining and Machine learning Chapter 13.3
"""

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def expectation_maximization(dataset, k, eps, max_iteration=100):
    """
    Method will call the main algorithm till the centroid converges i.e. the difference between
    two iterations for all centroids in less than or equal to Epsilon
    :param max_iteration: Max iterations to run
    :param dataset: input dataset as two dimensional np array
    :param k: Number of clusters
    :param eps: Epsilon to check convergence condition
    :return: data frame with probability for each point with the assigned cluster numbers
    """
    # find the number of dimensions from dataset
    d = dataset.shape[1]
    t = 0
    # assign variance to identity matrix
    sigma_i = np.identity(d)
    # initialize parameters probability for each cluster, centroids for each cluster and variances
    p_ci, mus, sigmas = init_para(k, d, sigma_i, dataset)
    while True:
        # this is used for counting iterations and stopping loop after max iterations
        t += 1
        # call the main algorithm, it returns
        # wij : weight each point on each clusetr
        # p_ci_new : update probability for each cluster
        # mus_new: updated centroids
        # updated variance matrix
        wij, p_ci_new, mus_new, sigmas_new = perform_iteration(dataset, p_ci, mus, sigmas)
        # print("mus:", mus)
        # Stopping condition, no of iterations or converged
        if t > max_iteration or (t > 1 and get_diff_mus(mus_new, mus, k) <= eps):
            break
        # Assign update variables
        p_ci = p_ci_new
        mus = mus_new
        sigmas = sigmas_new
    # print Results
    print("Final Mus:")
    for mu in mus:
        print(mu)
    print("Final Covariance:")
    for sigma in sigmas:
        print(sigma, end="\n")
    print("Number of Iterations:", t)
    df = pd.DataFrame(wij)
    df["cluster"] = df.apply(lambda row: find_max_prob(row), axis=1)
    print("Cluster assignments:")
    print(df)
    ax = sns.distplot(df.cluster, bins=k)
    plt.show()
    return df


def find_max_prob(row):
    """
    This method is applied over each row to find the cluster for each point by finding max probability
    :param row:
    :return: index+1 i.e. cluster of column with max probability
    """
    # find max probability
    max = row.max()
    # compare and find matching index
    for i in range(len(row)):
        if max == row[i]:
            return i + 1


def get_diff_mus(mus_new, mus, k):
    """
    Find euclidean distance between all centroids with old centroids points
    :param mus_new:
    :param mus:
    :param k:
    :return: maximum distance from centroids by comparing with last iteration
    """
    max_distance = 0
    for i in range(k):
        dist = find_distance(mus_new[i].reshape(mus_new[i].size), mus[i].reshape(mus_new[i].size))
        if max_distance < dist:
            max_distance = dist
    return max_distance


def find_distance(p1, p2):
    """
    Find euclidean distance between points
    :param p1:
    :param p2:
    :return:
    """
    return sum((i - j) * (i - j) for i, j in zip(p1, p2))


def init_para(k, d, sigmai, dataset):
    """
    Initialize parameters for algorithm
    :param k: number of clusters
    :param d: number of dimensions
    :param sigmai: sigma/variance matrix
    :param dataset: dataset
    :return: initialized centroids for each cluster and variances
    """
    # Calculate equal probability for each cluster
    ck = np.array(np.ones((k, 1))) * (1 / float(k))
    # initialize centroids randomly
    mus = initMus(d, k, dataset)
    sigmas = []
    # initialize variance with diagonal matrix of dXd
    for i in range(k):
        sigmas.append(sigmai)
    return ck, mus, sigmas


def calculate_cluster_probability(X, mu, variance):
    """
    Calculate the cluster probabilities by using
    :param X: xj the one row from dataset
    :param mu: Centroids for cluster
    :param variance: variance matrix for cluster
    :return: updated probability for cluster
    """
    dim = np.shape(X)[0]

    X = np.mat(X)
    mu = np.mat(mu)

    x_min_mu = X - mu
    variance = np.mat(variance)
    var_deter = np.linalg.det(variance)
    var_inv = np.linalg.pinv(variance)
    nom_exp_term = x_min_mu * var_inv * x_min_mu.transpose()
    prob = 1 / (((2 * math.pi) ** (dim / 2)) * (var_deter ** 0.5)) * np.exp(-0.5 * nom_exp_term)
    return prob


def perform_iteration(X, pt, mu, sigma):
    """
    This function will iterate over all points in dataset and will calculate probability for each point for each cluster
    :param X: Dataset
    :param pt: probability for all clusters
    :param mu: centroids for all clusters
    :param sigma: list of variance matrix for all clusters
    :return: wij: updated probabilities for each point, pi_new updated probabilities for each cluster, mu_new updated centroids
    , sigma_new updated variance
    """
    num, dim = np.shape(X)
    c, na = np.shape(pt)
    k = len(mu)
    prob_for_cluster = np.array(np.zeros((k, 1)))
    wij = np.array(np.zeros((num, k)))
    sigma_new = []
    mu_new = []
    pis_new = np.array(np.zeros((c, 1)))

    for i in range(num):
        total = 0
        for j in range(k):
            wij[i, j] = pt[j] * calculate_cluster_probability(X[i], mu[j], sigma[j])
            total += wij[i, j]
        # normalize probabilities to have sum as 1
        for j in range(k):
            wij[i, j] = wij[i, j] / total

    # updated centroids
    for j in range(k):
        tmp_mu = np.array(np.zeros((1, dim)))
        for i in range(num):
            prob_for_cluster[j, 0] += wij[i, j]
            tmp_mu += wij[i, j] * X[i]
        mu_new.append(tmp_mu / prob_for_cluster[j, 0])

    # updated variances
    for j in range(k):
        tmp_var = np.array(np.zeros((dim, dim)))
        for i in range(num):
            tmp_var += wij[i, j] * (
                    (X[i] - mu_new[j]).transpose() * (X[i] - mu_new[j]))
        sigma_new.append(tmp_var / prob_for_cluster[j, 0])

    # update probabilities for each cluster
    for j in range(k):
        pis_new[j] = prob_for_cluster[j] / num

    return wij, pis_new, mu_new, sigma_new


def initMus(d, k, dataset):
    """
    Initialize mus(centroids for clusters) for k clusters with d dimension for dataset
    :param d:
    :param k:
    :param dataset:
    :return: mus as data frame
    """
    data = pd.DataFrame(dataset)
    mins = data.min()
    maxs = data.max()
    mus = []
    for i in range(d):
        mus.append([xs for xs in np.random.uniform(mins[i], maxs[i], k)])
    mu = pd.DataFrame(mus)
    mu = mu.transpose()
    return np.array(mu)


def calculate_purity(wij, k):
    """
    Calculate purity for clusters
    :param wij: data frame with cluster assignments and labels
    :param k: number of clusters
    :return: returns purity
    """
    ti = np.array(wij.groupby(by="cluster").count()[0])
    ci = np.array(wij.groupby(by="label").count()[0])
    total_observations = 0
    for i in range(k):
        total_observations += min(ti[i], ci[i])
    purity = total_observations / wij.shape[0]
    return purity


def main():
    """
    Main method calling other methods
    :return:
    """
    dataset = pd.read_csv("iris.csv", header=None, names=["x1", "x2", "x3", "x4", "x5"])
    # number of clusters
    k = 3
    wij = expectation_maximization(np.array(dataset.iloc[:, [0, 1, 2, 3]]), k, 0.001)
    wij["label"] = dataset.x5
    print("Clusters Assigned by Algorithm:", "\n", wij.groupby(by="cluster").count()[0])
    print("Original Clusters:", "\n", wij.groupby(by="label").count()[0])
    print("Purity:", calculate_purity(wij, k))

    # dataset = np.array([[1.0], [1.3], [2.2], [2.6], [2.8], [5.0], [7.3], [7.4], [7.5], [7.7], [7.9]])
    # expectation_maximization(dataset, 2, 0.0001)


if __name__ == "__main__":
    ''' This will check if main method is called over the python file
    '''
    # call main function
    main()
