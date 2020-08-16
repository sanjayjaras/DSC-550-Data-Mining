"""
denclue.py

@author: mgarrett
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
import networkx as nx
import concurrent.futures as cf


def find_kernel_value(xt, xi, h, d):
    # kernel = np.exp(-(np.linalg.norm(xt - D[i]) / h) ** 2. / 2.) / ((2. * np.pi) ** (d / 2))
    # kernel = kernel * 1 / (h ** d)

    first_term = 1 / ((2 * np.pi) ** (d / 2))
    # second_term = ((xt - xi).transpose() - (xt - xi)) / (2 * h ** 2)
    # kernel = first_term * np.exp(-second_term)

    second_term_1 = np.linalg.norm(xt - xi) / (2 * h ** 2)
    kernel2 = first_term * np.exp(-second_term_1)

    return kernel2


def find_attractor(xt, D, h, eps):
    xt_plus_1 = np.copy(xt)
    t = 0.
    n = D.shape[0]
    d = D.shape[1]
    while True:
        xt = np.copy(xt_plus_1)
        influence = 0.
        xt_plus_1 = np.zeros((1, d))
        for i in range(n):
            kernel = find_kernel_value(xt, D[i], h, d)
            influence = influence + kernel
            xt_plus_1 = xt_plus_1 + (kernel * D[i])
        xt_plus_1 = xt_plus_1 / influence
        density = influence / n
        distance = np.linalg.norm(xt_plus_1 - xt)
        t += 1
        if distance > eps:
            break
    return [xt_plus_1, density]


def find_attractors_mt(D, h, eps):
    n = D.shape[0]
    d = D.shape[1]
    density_attractors = np.zeros((n, d))
    densities = np.zeros((n, 1))

    dummy_D = [D for _ in range(n)]
    dummy_h = [h for _ in range(n)]
    dummy_eps = [eps for _ in range(n)]

    with cf.ProcessPoolExecutor() as executor:
        results = executor.map(find_attractor, D, dummy_D, dummy_h, dummy_eps)

    i = 0
    for result in results:
        density_attractors[i] = result[0]
        densities[i] = result[1]
        i += 1
    print("Done with finding density attractors....")
    return density_attractors, densities


class DENCLUE(BaseEstimator, ClusterMixin):
    """Perform DENCLUE clustering from vector array.

    Parameters
    ----------
    h : float, optional
        The smoothing parameter for the gaussian kernel. This is a hyper-
        parameter, and the optimal value depends on data. Default is the
        np.std(X)/5.

    eps : float, optional
        Convergence threshold parameter for density attractors

    min_density : float, optional
        The minimum kernel density required for a cluster attractor to be
        considered a cluster and not noise.  Cluster info will stil be kept
        but the label for the corresponding instances will be -1 for noise.
        Since what consitutes a high enough kernel density depends on the
        nature of the data, it's often best to fit the model first and
        explore the results before deciding on the min_density, which can be
        set later with the 'set_minimum_density' method.
        Default is 0.

    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. In this version, I've only tested 'euclidean' at this
        moment.

    Attributes
    -------
    cluster_info_ : dictionary [n_clusters]
        Contains relevant information of all clusters (i.e. density attractors)
        Information is retained even if the attractor is lower than the
        minimum density required to be labelled a cluster.

    labels_ : array [n_samples]
        Cluster labels for each point.  Noisy samples are given the label -1.

    Notes
    -----


    References
    ----------
    Hinneburg A., Gabriel HH. "DENCLUE 2.0: Fast Clustering Based on Kernel
    Density Estimation". In: R. Berthold M., Shawe-Taylor J., Lavrač N. (eds)
    Advances in Intelligent Data Analysis VII. IDA 2007
    """

    def __init__(self, h=None, eps=1e-8, min_density=0., metric='euclidean'):
        self.h = h
        self.eps = eps
        self.min_density = min_density
        self.metric = metric

    def fit(self, X, y=None):
        if not self.eps > 0.0:
            raise ValueError("eps must be positive.")
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        # create default values
        if self.h is None:
            self.h = np.std(X) / 5

        # initialize all labels to noise
        labels = -np.ones(X.shape[0])

        density_attractors, density = find_attractors_mt(X, self.h, self.eps)
        print("Done with finding density attractors....")
        # climb each hill
        # for i in range(self.n_samples):
        #    density_attractors[i], density[i], radii[i] = _hill_climb(X[i], X, h=self.h, eps=self.eps)

        # initialize cluster graph to finalize clusters. Networkx graph is
        # used to verify clusters, which are connected components of the
        # graph. Edges are defined as density attractors being in the same
        # neighborhood as defined by our radii for each attractor.
        cluster_info = {}
        num_clusters = 0
        cluster_info[num_clusters] = {'instances': [0],
                                      'centroid': np.atleast_2d(density_attractors[0])}
        g_clusters = nx.Graph()

        for j1 in range(self.n_samples):
            g_clusters.add_node(j1, attr_dict={'attractor': density_attractors[j1],
                                               'density': density[j1]})
        print("Completed creating graph...")
        # populate cluster graph
        for j1 in range(self.n_samples):
            for j2 in (x for x in range(self.n_samples) if x != j1):
                if g_clusters.has_edge(j1, j2):
                    continue
                # print(g_clusters.node[j1])
                diff = np.linalg.norm(
                    g_clusters.node[j1]['attr_dict']['attractor'] - g_clusters.node[j2]['attr_dict']['attractor'])
                # if diff <= (g_clusters.node[j1]['attr_dict']['radius']+g_clusters.node[j1]['attr_dict']['radius']):
                if diff <= self.eps:
                    g_clusters.add_edge(j1, j2)

        # connected components represent a cluster
        clusters = list(nx.connected_component_subgraphs(g_clusters))
        num_clusters = 0

        # loop through all connected components
        for clust in clusters:

            # get maximum density of attractors and location
            max_instance = max(clust, key=lambda x: clust.node[x]['attr_dict']['density'])
            max_density = clust.node[max_instance]['attr_dict']['density']
            max_centroid = clust.node[max_instance]['attr_dict']['attractor']

            # In Hinneberg, Gabriel (2007), for attractors in a component that
            # are not fully connected (i.e. not all attractors are within each
            # other's neighborhood), they recommend re-running the hill climb
            # with lower eps. From testing, this seems unnecesarry for all but
            # special edge cases. Therefore, completeness info is put into
            # cluster info dict, but not used to re-run hill climb.
            complete = False
            c_size = len(clust.nodes())
            if clust.number_of_edges() == (c_size * (c_size - 1)) / 2.:
                complete = True

            # populate cluster_info dict
            cluster_info[num_clusters] = {'instances': clust.nodes(),
                                          'size': c_size,
                                          'centroid': max_centroid,
                                          'density': max_density,
                                          'complete': complete}

            # if the cluster density is not higher than the minimum,
            # instances are kept classified as noise

            ################################################
            # if max_density[0] >= self.min_density:
            #     labels[clust.nodes()] = num_clusters
            num_clusters += 1

        self.clust_info_ = cluster_info
        self.labels_ = labels
        return self

    def get_density(self, x, X, y=None, sample_weight=None):
        superweight = 0.
        n_samples = X.shape[0]
        n_features = X.shape[1]
        if sample_weight is None:
            sample_weight = np.ones((n_samples, 1))
        else:
            sample_weight = sample_weight
        for y in range(n_samples):
            kernel = kernelize(x, X[y], h=self.h, degree=n_features)
            kernel = kernel * sample_weight[y] / (self.h ** n_features)
            superweight = superweight + kernel
        density = superweight / np.sum(sample_weight)
        return density

    def set_minimum_density(self, min_density):
        self.min_density = min_density
        labels_copy = np.copy(self.labels_)
        for k in self.clust_info_.keys():
            if self.clust_info_[k]['density'] < min_density:
                labels_copy[self.clust_info_[k]['instances']] = -1
            else:
                labels_copy[self.clust_info_[k]['instances']] = k
        self.labels_ = labels_copy
        return self


if __name__ == "__main__":
    """
    This will check if main method is called over the python file
    """

    dataset = pd.read_csv("iris.txt", header=None, names=["x1", "x2", "x3", "x4", "x5"])

    denclue = DENCLUE(min_density=0.0005, eps=0.0001, h=39.7)
    denclue.fit(np.array(dataset.iloc[:, [0, 1, 2, 3]]))
    print(denclue)
    print(len(denclue.clust_info_))
