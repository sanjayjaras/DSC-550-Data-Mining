import concurrent.futures as cf
import random
from collections import Counter

import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

plt.style.use("seaborn-darkgrid")
plt.rcParams["figure.figsize"] = [24, 12]


def diagonals_in_hd():
    """
    Find random corners for hypercubes with 10, 100 and 1000 dimensions and find the angle between them
    also show the information about angles for each dimension
    :return:
    """
    number_of_pairs = 100000
    angles_for_d = {}
    for d in (10, 100, 1000):
        number_of_corners = 2 ** d - 1
        first_corner = [random.randint(0, number_of_corners) for _ in range(0, number_of_pairs)]
        second_corner = [random.randint(0, number_of_corners) for _ in range(0, number_of_pairs)]

        dummy_d = [d for _ in range(0, number_of_pairs)]
        angles = []
        with cf.ProcessPoolExecutor() as executor:
            results = executor.map(find_angle, first_corner, second_corner, dummy_d)
        for result in results:
            angles.append(result)
        ser = pd.Series(angles)
        print(f"Angles between diagonals for {d} dimensions")
        print(ser.describe())
        angles_for_d[d] = ser

    plot_pmfs_for_ds(angles_for_d)


def plot_pmfs_for_ds(angles_for_d):
    """
    Plot pmfs for dimensions in dictionary
    :param angles_for_d:
    :return:
    """
    for key, value in angles_for_d.items():
        pmf = Pmf(value)
        pmf.normalize()
        xs, ys = pmf.render()
        ser_p = pd.Series(xs) * pd.Series(ys)
        mu = ser_p.sum()
        print(f"Expected Theta is:{mu} for dimension:{key}")
        plt.plot(xs, ys, label=key)
    plt.xticks(range(0, 180, 5))
    plt.legend()
    plt.show()


class Pmf(Counter):
    """A Counter with probabilities."""

    def normalize(self):
        """Normalizes the PMF so the probabilities add to 1."""
        total = float(sum(self.values()))
        for key in self:
            self[key] /= total

    def __add__(self, other):
        pmf = Pmf()
        for key1, prob1 in self.items():
            for key2, prob2 in other.items():
                pmf[key1 + key2] += prob1 * prob2
        return pmf

    def __hash__(self):
        """Returns an integer hash value."""
        return id(self)

    def __eq__(self, other):
        return self is other

    def render(self):
        """Returns values and their probabilities, suitable for plotting."""
        return zip(*sorted(self.items()))


def decimal_to_corner_vector(n, number_of_bits):
    """
    Convert number into fix size series for hypercube corner with dimensions in -1, 1
    :param n: number represent corner
    :param number_of_bits: represent number of dimensions
    :return: series with corner coordinates
    """
    s = bin(n).replace("0b", "")
    s = s.rjust(number_of_bits, '0')
    c1 = pd.Series(list(s))
    c1 = c1.astype(int)
    return c1.replace(0, -1)


def find_angle(corner1, corner2, d):
    """
    Find angle between the diagonals represented with corners for d dimensional hypercube
    :param corner1:
    :param corner2:
    :param d:
    :return:
    """
    c1 = decimal_to_corner_vector(corner1, d)
    c2 = decimal_to_corner_vector(corner2, d)
    c1_magn = np.sqrt(np.square(c1).sum())
    c2_magn = np.sqrt(np.square(c2).sum())
    dot = c1.dot(c2)
    magn_prod = c1_magn * c2_magn
    res = math.degrees(math.acos(dot / magn_prod))
    return res


def main():
    """
    Maim method that will call other methods
    :return:
    """
    diagonals_in_hd()


# Check for main is called on this file
if __name__ == "__main__":
    ''' This will check if main method is called over the python file
    '''
    print("Processing...")
    # call main function
    main()
    print("Please see additional information in attached document.")
