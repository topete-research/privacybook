import pandas as pd

def max_kanonymity(D):
    '''Outputs the maximum value of k for which D represents a k-anonymous dataset. D is
    a pandas dataframe created using the make_channel function.'''
    return D.astype(bool).sum(axis=0).min()

def is_kanonymous(D, k):
    '''Outputs true if the dataset D satisfies k-anonymity. D is a pandas dataframe
    created using the make_channel function.'''
    max_k = max_kanonymity(D)
    return max_k >= k

def vulnerable_identifiers(D, k):
    '''Outputs the column names for which k-anonymity does not hold.'''
    # k_values contains the k-anonymity value for each column
    k_values = D.astype(bool).sum(axis=0)
    # return the column names and k-anonymity values for which k_values are < k.
    return k_values[k_values < k]
