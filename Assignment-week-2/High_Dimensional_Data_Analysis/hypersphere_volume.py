import math
import matplotlib.pyplot as plt

plt.style.use("seaborn-darkgrid")
plt.rcParams["figure.figsize"] = [24, 12]


def calculate_hypersphere_volumes(radius=1, from_dim=1, to_dim=50):
    """
    Calculate volumes for hypersphere for given range of dimensions and radius
    :param radius:
    :param from_dim:
    :param to_dim:
    :return: list of hypersphere volumes
    """
    volumes = []
    for d in range(from_dim, to_dim + 1):
        kd = calculate_hypersphere_volume(d, radius)
        volumes.append(kd)
    return volumes


def calculate_hypersphere_volume(d, radius):
    """
    Calculate volume for hypersphere for a dimension and radius
    :param d:
    :param radius:
    :return: volume of hypersphere
    """
    r_pow_d = radius ** d
    kd = calculate_kd(d)
    vol = r_pow_d * kd
    return vol


def calculate_kd(d):
    """
    Calculate the value of Kd required to calculate volume of hypersphere
    formula for Hyperspehere volume = Kd.r^d  where Kd= Pi^(d/2) / r(d/2+1)
    Here r(d/2+1) is d/2 ! when d is even sqrt(Pi) * (d!!/2^(d+1)/2) when d is odd
    d!! is 1 whene d=0 or d==1 and d.(d-2)!! if d>=1
    :param d:
    :return:
    """

    pi_pow_d_by_2 = math.pi ** (d / 2)
    if d % 2 == 1:  # d is odd
        two_pow_d_plus1_by_2 = 2 ** ((d + 1) / 2)
        r_d_by_2_plus_1 = math.sqrt(math.pi) * (doublefactorial(d) / two_pow_d_plus1_by_2)
    else:  # even
        r_d_by_2_plus_1 = math.factorial(d / 2)
    kd = pi_pow_d_by_2 / r_d_by_2_plus_1
    return kd


def doublefactorial(n):
    """
    Calculate double factorial
    :param n:
    :return:
    """
    if n < 2:
        return 1
    else:
        return n * doublefactorial(n - 2)


def plot_hypersphere_volume(volumes):
    """
    Plot the line graph for volumes of unit hypersphere for volumes in list
    :param volumes:
    :return:
    """

    plt.plot(range(1, len(volumes) + 1), volumes, marker="o")
    plt.xticks(range(0, len(volumes) + 1, 2))
    plt.title("Volume for unit Hypersphere")
    plt.xlabel("d")
    plt.ylabel("vol(Sd(1))")
    plt.show()


def main():
    """
    Maim method that will call other methods
    :return:
    """
    # call volume calculations for range
    volumes = calculate_hypersphere_volumes(radius=1, from_dim=1, to_dim=50)
    # plot the graph for volumes
    plot_hypersphere_volume(volumes)


# Check for main is called on this file
if __name__ == "__main__":
    ''' This will check if main method is called over the python file
    '''
    print("Processing...")
    # call main function
    main()
