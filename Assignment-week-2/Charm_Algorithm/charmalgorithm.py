import pandas as pd
import sys


def create_dict_from_file(filename):
    """ Read in a file of item sets each row is considered the transaction id and each line contains the items associated with it.
        This function returns a dictionary that has a key set as the tid and has values of the list of items (integers)
    """
    d = {}
    with open(filename, 'r') as f:
        for tids, line_items in enumerate(f):
            d[tids] = [num.strip() for num in line_items.split(' ') if num != '\n']
    return d


def create_database(item_set):
    "Uses dummy indexing to create the binary database"
    # Convert dictionary to series with list as elements
    sr = pd.Series(item_set)
    # convert list elements to string separated by |
    sr = sr.str.join("|")
    # convert categorical variables to dummy columns with each item(number) as column
    return sr.str.get_dummies()


def compute_support(df):
    "Exploits the binary nature of the database"
    # sum all the column as values are in binary form
    return df.sum()


def filter_as_per_support(sr, min_support):
    """
    Filter the items from series those doesn't match minimum support criteria
    :param sr:
    :param min_support:
    :return: new list after filtering
    """
    return [(index, value) for index, value in sr.items() if value >= min_support]


def string_union(s1, s2, sep="|"):
    """
    Perform Union of 2 strings
    :param s1: string with item names(Xi) separator
    :param s2: string with item names(Xj) separator
    :param sep: separator default |
    :return: return concatenated string of item names by removing duplicate elements
    """
    result = s1
    s1_keys = s1.split(sep)
    for key in s2.split(sep):
        if key not in s1_keys:
            result += "|" + key
    return result


def is_string_subset(s, b, sep="|"):
    """
    check if s is subset of b
    :param s: item name list with separator
    :param b: item name list with separator
    :param sep: Separator
    :return: boolean True if s is subset ob b
    """
    b_keys = b.split(sep)
    for s_key in s.split(sep):
        if s_key not in b_keys:
            return False
    return True


def replace_column(df, old_series_name, new_series_name, new_series):
    """
    Replace column in data frame and rename as well
    :param df:
    :param old_series_name:
    :param new_series_name:
    :param new_series:
    :return: data frame with replaced and renamed column
    """
    df[old_series_name] = new_series
    return df.rename(columns={old_series_name: new_series_name})


def charm(p, min_support, c):
    """
    Execute the Mining Closed Frequent Item sets: Charm Algorithm on data frame p, with minimum support
    and c will contain the closed frequent items sets
    :param p:
    :param min_support:
    :param c:
    :return:
    """
    # print(ip.columns)
    i = 0
    while i < len(p.columns):  # foreach hX i , t(X i )i ∈ P do
        pi = pd.DataFrame()  # P i ← ∅
        j = i + 1
        # foreach hX j , t(X j )i ∈ P , with j > i do
        while j < len(p.columns):
            # Xi Union Xj
            Xij = string_union(p.columns[i], p.columns[j])
            # t(Xi) intersection t(Xj), binary column so it will be 1 when both columns are 1
            tXij = p[p.columns[i]] & p[p.columns[j]]
            # if sup(X ij ) ≥ minsup then
            if tXij.sum() >= min_support:
                # print(f"Support{tXij.sum()}")
                # if t(X i ) = t(X j ) then // Property 1
                if (p[p.columns[i]] == p[p.columns[j]]).all():
                    # Replace X i with X ij in P and P i
                    pi = replace_column(pi, p.columns[i], Xij, tXij)
                    p = replace_column(p, p.columns[i], Xij, tXij)
                    # Remove hX j , t(X j )i from P
                    del p[p.columns[j]]
                else:
                    no_of_1_Xi = p[p.columns[i]].sum()
                    # checking if p[Xi] is subset of p[Xj] Property 2
                    if (p[p.columns[i]] * p[p.columns[j]]).sum() == no_of_1_Xi:
                        # Replace X i with X ij in P and P i
                        pi = replace_column(pi, p.columns[i], Xij, tXij)
                        p = replace_column(p, p.columns[i], Xij, tXij)
                    else:  # Property 3
                        pi[Xij] = tXij
            j += 1
        # if Pi is not empty then CHARM(Pi, minsup, C)
        if len(pi.columns) != 0:
            charm(pi, min_support, c)
        found_in_c = check_itemset_already_added(c, i, p)
        # if X i is not a subset of any closed set Z with the same support
        if not found_in_c:
            # print(f"Adding new set to C:{p.columns[i]} and support:{p[p.columns[i]].sum()}")
            # C = C union Xi // Add X i to closed set
            c.append((p.columns[i], p[p.columns[i]].sum()))
        i += 1


def check_itemset_already_added(existing_set, column_index, data_frame):
    """
     Check if item set already added to closed frequent item sets by checking subset and support
    :param existing_set:
    :param column_index:
    :param data_frame:
    :return:
    """
    found_in_c = False
    for tup in existing_set:
        if is_string_subset(data_frame.columns[column_index], tup[0]) \
                and data_frame[data_frame.columns[column_index]].sum() == tup[1]:
            found_in_c = True
            break
    return found_in_c


def main(fileName, minsup):
    """
    Main method that will call other methods
    :param fileName:
    :param minsup:
    :return:
    """
    # create dictionary by reading the file
    dictionary = create_dict_from_file(fileName)
    # create data frame with binary variables for each item
    db_df = create_database(dictionary)
    # calculate support for each item
    sums = compute_support(db_df)
    # filter items by minimum support criteria
    filterd_sup = filter_as_per_support(sums, minsup)
    # sort items by increasing order of support
    filterd_sup.sort(key=lambda x: x[1])
    # create data frame by using support order from above
    df_gt_sup = pd.DataFrame()
    for tup in filterd_sup:
        df_gt_sup[tup[0]] = db_df[tup[0]]

    # create place holder for closed frequent item sets
    c = []
    # call charm algorithm method
    charm(df_gt_sup, minsup, c)
    print("closed itemset - support")
    for ci in c:
        print(ci)


# Check for main is called on this file
if __name__ == "__main__":
    ''' This will check if main method is called over the python file
    '''

    # Check if the command line arguments are given
    try:
        print('Filename: ', sys.argv[1])
        print('Min Support: ', sys.argv[2])
    except:
        print('You need both a filename and a minimum support value!')
        print("Usage: charmalgorithm <file-name> <minsup>")
        print("e.g.: charmalgorithm mashroom.txt 3000")
    minsup = int(sys.argv[2])

    # call main function
    main(sys.argv[1], minsup)
