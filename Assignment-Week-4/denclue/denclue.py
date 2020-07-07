"""
Program to implement Denclue density based algorithm  from the book
Data Mining and Machine learning Chapter 15.2
"""
import pandas as pd
import numpy as np
import concurrent.futures as cf


def find_kernel_value(xt, xi, h, d):
    """
    Calculate Gaussian kernel value
    :param xt: the center point
    :param xi: target point
    :param h: window size
    :param d: number of features/dimension
    :return: gaussian kernel value
    """
    first_term = 1 / ((2 * np.pi) ** (d / 2))
    second_term_1 = np.linalg.norm(xt - xi) / (2 * h ** 2)
    kernel2 = first_term * np.exp(-second_term_1)
    return kernel2


def find_attractor(x, D, h, eps):
    """
    Find density attractor by climbing hill by using gaussian kernel
    :param x: the point for which we want to find attracter
    :param D: Dataset
    :param h: window size
    :param eps: Epsilon value; tolerance for convergence
    :return: density attractor point
    """
    xt = x
    d = D.shape[1]
    while True:
        nom = 0
        den = 0
        for xi in D:
            kernel = find_kernel_value(xt, xi, h, d)
            nom += kernel * xi
            den += kernel
        xt_plus_1 = nom / den
        if np.linalg.norm(xt - xt_plus_1) > eps:
            return np.round(xt_plus_1, decimals=1)
        else:
            xt = xt_plus_1


def find_density(attr, D, h):
    """
    Find density for point by using gaussian kernel
    :param attr: point to calculate density for
    :param D: Dataset
    :param h: window size
    :return: returns density for point
    """
    d = D.shape[1]
    n = D.shape[0]
    total = 0
    for xi in D:
        kernel = find_kernel_value(attr, xi, h, d)
        total += kernel
    return total / (n * h ** d)


def find_attractors_mt(D, h, eps):
    """
    Wrapper method to apply multi-threading for convergence
    :param D: Dataset
    :param h: window size
    :param eps: epsilon, tolerance for convergence
    :return: returns all density attractors for all points in dataset
    """
    n = D.shape[0]
    d = D.shape[1]
    density_attractors = np.zeros((n, d))

    dummy_D = [D for _ in range(n)]
    dummy_h = [h for _ in range(n)]
    dummy_eps = [eps for _ in range(n)]

    with cf.ProcessPoolExecutor() as executor:
        results = executor.map(find_attractor, D, dummy_D, dummy_h, dummy_eps)

    i = 0
    for result in results:
        density_attractors[i] = result
        i += 1

    return density_attractors


def find_maximal_subset(clusters, attractor, R):
    """
    Method to find maximal clusters step number 7 from algorithm
    :param clusters: already created clusters
    :param attractor: attractor to be added in appropriate cluster
    :param R: all the points set by attractor
    :return: returns updated clusters
    """
    # TODO: implement maximal subset logic if new attractor is density reachable by all existing attractors in cluster add that point to cluster
    """ dens_reach = False
     for cl in clusters:
         dens_reach = True
         for ex_att in cl:
             dist = np.linalg.norm(ex_att - attractor)
             if dist > 10:
                 dens_reach = False

         if (dens_reach):
             cl.append(attractor)
             break
     if not dens_reach:
         clusters.append([attractor])
     """
    clusters.append([attractor])


def density_based_cluster(R, clusters):
    """
    Assign points to appropriate clusters
    :param R: map of attractor and assigned points
    :param clusters: cluster with attractors
    :return: cluster map with points
    """
    c_points = {}
    i = 0
    for cluster in clusters:
        points = set()
        for attr in cluster:
            for point in R[attr]:
                points.add(point)
        c_points[i] = points
        i += 1
    return c_points


def calculate_purity(D, k):
    """
    Calculate purity for clusters
    :param D: data frame with cluster assignments and labels
    :param k: number of clusters
    :return: returns purity
    """
    ti = np.array(D.groupby(by="cluster").count()['x1'])
    ci = np.array(D.groupby(by="label").count()['x1'])
    total_observations = 0
    for i in range(k):
        total_observations += min(ti[i], ci[i])
    purity = total_observations / D.shape[0]
    return purity


def denclue(D, min_dens, eps, h):
    """
    Denclue algorithm method
    :param D: Dataset
    :param min_dens: Minimum density threshold
    :param eps: epsilon tolerance for convergence
    :param h: window size
    :return: clusters with assigned points
    """
    density_attractors = find_attractors_mt(D, h, eps)
    A = set()
    R = {}
    i = 0
    for x in D:
        attr = density_attractors[i]
        if find_density(attr, D, h) >= min_dens:
            A.add(tuple(attr))
            if tuple(attr) not in R:
                R[tuple(attr)] = set()
            R[tuple(attr)].add(i)

        i += 1
    clusters = []
    for att in A:
        find_maximal_subset(clusters, att, eps)

    c_points = density_based_cluster(R, clusters)

    print(f"No of clusters:{len(c_points)}")


    return c_points, clusters


def main(dataset=None, min_dens=1 * 10 ** -6, eps=0.0001, h=7):
    """
    Main method to call denclue algorithm
    :param dataset: dataset
    :param min_dens: Minimum density threshold
    :param eps: epsilon tolerance for convergence
    :param h: Window size
    :return:
    """
    if dataset is None:
        dataset = pd.read_csv("iris.txt", header=None, names=["x1", "x2", "x3", "x4", "label"])
    print(f"Inputs mindensity:{min_dens}\teps:{eps}\th:{h}")
    cluster_map, cluster_center = denclue(np.array(dataset.iloc[:, [0, 1, 2, 3]]), min_dens, eps, h)
    dataset["cluster"] = -1
    for c, points in cluster_map.items():
        dataset.at[points, "cluster"] = c
        print("Attractor:", cluster_center[c])
        print("Points in cluster:", points)
    print("Size of each cluster")
    print("Clusters Assigned by Algorithm:", "\n", dataset.groupby(by="cluster").count()['x1'])
    print("Original Clusters:", "\n", dataset.groupby(by="label").count()['x1'])
    print("Purity:", calculate_purity(dataset, len(cluster_map)))


if __name__ == "__main__":
    """
    This will check if main method is called over the python file
    """
    main(h=7, min_dens=1 * 10 ** -6)
