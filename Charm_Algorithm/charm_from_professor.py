""" This algorithm replicates the Charm algorithm
     from Chapter 9 in Data Science and Machine
     learning.
"""
import sys
import pandas as pd


def create_dict_from_file(filename):
    """ Read in a file of itemsets
        each row is considered the transaction id
        and each line contains the items associated
        with it.
        This function returns a dictionary that
        has a key set as the tid and has values
        the list of items (strings)
    """
    f = open(filename, 'r')
    d = {}
    for tids, line_items in enumerate(f):
        d[tids] = [j.strip('\n') for j in line_items.split(' ')
                   if j != '\n']
    return d


def create_database(itemset):
    "Uses dummy indexing to create the binary database"
    return pd.Series(itemset).str.join('|').str.get_dummies()


def create_initial_list(database):
    # Returns a list of each column name with the tids it appears
    base_list = []
    for col in database.columns:
        base_list.append(([col], list(database[database[col] == 1
                                               ].index.values)))
    return base_list


def join_items(a_list, b_list):
    """ This function returns the unique union of two
        lists elements as a list
    """
    return list(set(a_list + b_list))


def join_tids(a_list, b_list):
    """ This function returns the intersection of two
        sets of tids as a list
    """
    return list(set(a_list).intersection(b_list))


def list_equal(a_list, b_list):
    """ This function returns if the two lists of 
        tids are equal
    """
    return (len(a_list) == len(b_list) and set(a_list) == set(b_list))


def list_contained(a_list, b_list):
    # This function checks if a is contained in b
    return all([element in b_list for element in a_list])


def check_closed(tup_item, closed_list):
    """ Check if the tup_item (itemset,tids) meets two
        conditions: if itemset isn't a subset of a 
        any previous closed itemset AND the tids are not
        the same
    """
    if not closed_list:
        return True
    for itemset, tids in closed_list:
        if (list_contained(tup_item[0], itemset) and
                list_equal(tup_item[1], tids)):
            return False
    return True


def find_replace_items(find_list, replace_list, ref_list):
    return_list = []
    for itemset, tids in ref_list:
        if list_contained(find_list, itemset):
            new_items = list(set(itemset + replace_list))
        else:
            new_items = itemset
        return_list.append((new_items, tids))
    return return_list


def charm(p_list, minsup, c_list):
    """ This is the implementation of charm
        where we are passed:
        p_list: a list of (itemset, tids)
        minsup: a parameter of the freq threshold
        c_list: closed list of itemsets
    """
    # Sort the p_list in increasing support
    sorted_p_list = sorted(p_list, key=lambda
        entry: len(entry[1]))
    for i in range(len(sorted_p_list)):
        p_temp = []
        for j in range(i + 1, len(sorted_p_list)):
            if sorted_p_list[j] == ([], []): pass
            joined_items = join_items(sorted_p_list[i][0],
                                      sorted_p_list[j][0])
            joined_tids = join_tids(sorted_p_list[i][1],
                                    sorted_p_list[j][1])
            if len(joined_tids) >= minsup:
                if list_equal(sorted_p_list[i][1],
                              sorted_p_list[j][1]):
                    sorted_p_list[j] = ([], [])
                    temp_items = sorted_p_list[i][0]
                    sorted_p_list = find_replace_items(temp_items,
                                                       joined_items,
                                                       sorted_p_list)
                    p_temp = find_replace_items(temp_items,
                                                joined_items,
                                                p_temp)
                else:
                    if list_contained(sorted_p_list[i][1],
                                      sorted_p_list[j][1]):
                        temp_items = sorted_p_list[i][0]
                        sorted_p_list = find_replace_items(temp_items,
                                                           joined_items,
                                                           sorted_p_list)
                        p_temp = find_replace_items(temp_items,
                                                    joined_items,
                                                    p_temp)
                    else:
                        p_temp.append((joined_items, joined_tids))
        if p_temp:
            charm(p_temp, minsup, c_list)
        if check_closed(sorted_p_list[i], c_list):
            c_list.append(sorted_p_list[i])


if __name__ == '__main__':
    # Check if the command line arguments are given
    try:
        print('Filename: ', sys.argv[1])
        print('Min Support: ', sys.argv[2])
    except:
        print('You need both a filename and minimum support value!')

    minsup = int(sys.argv[2])
    dict_itemset = create_dict_from_file(sys.argv[1])
    database = create_database(dict_itemset)
    base_item_list = create_initial_list(database)
    filtered_list = [(item, tids) for item, tids in base_item_list
                     if len(tids) >= minsup]
    closed_sets = []

    # execute charm
    charm(filtered_list, minsup, closed_sets)

    # Sort and print. Filter out the empty set
    closed_sets = sorted([(i, j) for i, j in closed_sets if i != []], key=lambda x: x[1])
    for entry in closed_sets:
        print(entry[0], len(entry[1]))
