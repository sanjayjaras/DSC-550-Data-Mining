import hypersphere_volume as hv
import matplotlib.pyplot as plt

plt.style.use("seaborn-darkgrid")
plt.rcParams["figure.figsize"] = [24, 12]


def calculate_hypersphere_radius(d, volume):
    """
    Calculate radius for hypersphere with given dimension and volume
    :param d:
    :param volume:
    :return:
    """
    return (volume / hv.calculate_kd(d)) ** (1 / d)


def calculate_radius(from_dim=1, to_dim=50, volume=1):
    """
    Calculate hypersphere radius's for dimension range and volume
    :param volume:
    :param from_dim:
    :param to_dim:
    :return:
    """
    rs = []
    for d in range(from_dim, to_dim + 1):
        rs.append(calculate_hypersphere_radius(d, volume))
    return rs


def plot_hypersphere_radius(rs):
    """
    Plot the line graph for radius for list
    :param rs:
    :return:
    """

    plt.plot(range(1, len(rs) + 1), rs, marker="o")
    plt.xticks(range(0, len(rs) + 1, 2))
    plt.title("Radius of Hypersphere with Volume 1")
    plt.xlabel("d")
    plt.ylabel("radius")
    plt.show()


def main():
    """
    Maim method that will call other methods
    :return:
    """
    # calculate radius's for volumes and range
    rs = calculate_radius(from_dim=1, to_dim=50, volume=1)
    # plot hyper sphere radius
    plot_hypersphere_radius(rs)


# Check for main is called on this file
if __name__ == "__main__":
    ''' This will check if main method is called over the python file
    '''
    print("Processing...")
    # call main function
    main()
