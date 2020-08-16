import pandas as pd
import sys
from itertools import combinations


def create_dict_from_file(filename):
    """
     Read itemsets and their respective support into dictionary. itemsets tuple as key and support as value
    :param filename:
    :return: dictionary with itemsets tuple as key and support as value
    """
    d = {}
    with open(filename, 'r') as f:
        for line in f:
            items_support = line.split("-")
            items = tuple(item.strip() for item in items_support[0].split())
            support = items_support[1].strip()
            d[items] = support
    return d


def read_itemsets_from_file(filename):
    """
    Read Itemsets as tuples in list for which we will be finding lower and upper bounds
    :param filename:
    :return: Returns list of itemsets tuples
    """
    with open(filename, 'r') as f:
        return [(tuple(item.strip() for item in line.split())) for line in f]


def find_bounds_for_all(itemsets, item_supprt):
    """
    Find upper and lower bounds  also compare minimum and maximum for bounds to conclude if itemset is derivable or not
    :param itemsets:
    :param item_supprt:
    :return: dictionary key: tuple of itemset and value tuple consists of 2 tuples, lower/upper bound and conclusion
    """
    bounds = {}
    for itemset in itemsets:
        lu = find_bounds(itemset, item_supprt)
        bounds[itemset] = (lu, "Nonderivable" if max(lu[0]) != min(lu[1]) else "Derivable")
    return bounds


def find_bounds(itemset, itemsets_support):
    """
    Find upper and lower bounds
    :param itemset:
    :param itemsets_support:
    :return: tuple with list of lower bound and upper bounds
    """
    lower_bounds = []
    upper_bounds = []
    xlen = len(itemset)
    # Find all y values for itemset
    ys = get_all_y(itemset)
    # find all w's for each ys
    y_w_dictionary = find_ws(ys)
    # processing all w's for y
    for y, ws in y_w_dictionary.items():
        # decide upper bound or lower len(x) - len(y) if it's odd-> upper
        upper = False if (xlen - len(y)) % 2 == 0 else True  # even
        sum = 0
        for w in ws:
            # find multiplication factor or sign of support  -1 ^ (len(x)-len(y)+1)
            mult = (-1) ** (xlen - len(w) + 1)
            # cumulative sum of supports
            sum += mult * (len(ws) - 1 if w == () else get_support(w, itemsets_support))
        if upper:
            upper_bounds.append(sum)
        else:
            lower_bounds.append(sum)
    return (set(lower_bounds), set(upper_bounds))


def get_support(itemset, itemsets_support):
    """
    Find support from dataset
    :param itemset:
    :param itemsets_support:
    :return:
    """
    try:
        return int(itemsets_support[itemset])
    except KeyError as e:
        print(e)
        return 0


def find_ws(ys):
    """
    Find all w's from itemsets
    :param ys:
    :return:
    """
    d = {}
    for y in ys:
        d[y] = [itemset for itemset in ys if set(y).issubset(itemset)]
    return d


def get_all_y(itemset):
    """
    Find all possible combinations i.e. y's from itemset
    :param itemset:
    :return:
    """
    result = []
    for num_of_items in range(len(itemset)):
        result = result + (list(combinations(itemset, num_of_items)))
    return result


def main(itemset_support_file, itemset_file):
    """
    Main method that will call other methods
    :param itemset_support_file:
    :param itemset_file:
    :return:
    """
    # create dictionary by reading the file
    dictionary = create_dict_from_file(itemset_support_file)
    # create list of item sets from file
    itemsets = read_itemsets_from_file(itemset_file)
    # call the method that will calculate bounds and concluded from derivable or not
    bounds = find_bounds_for_all(itemsets, dictionary)
    # print Results
    for key, value in bounds.items():
        print(f"Itemset:{key}\tLower Bounds:{value[0][0]}\tUpper Bounds:{value[0][1]}\t{value[1]}")


# Check for main is called on this file
if __name__ == "__main__":
    ''' This will check if main method is called over the python file
    '''

    # Check if the command line arguments are given
    try:
        print('File Name 1: ', sys.argv[1])
        print('File Name 2: ', sys.argv[2])
    except:
        print('You need both a File Name 1 and File Name 2 value!')
        print("Usage: nonderivableitemset <file-name-1> <file-name-2>")
        print("e.g.: nonderivableitemset itemsets.txt ndi.txt")

    # call main function
    main(sys.argv[1], sys.argv[2])
