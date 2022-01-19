import numpy as np
from sklearn.metrics import mutual_info_score

"""
MIN_SAMPLE_SIZE:
The minimum number of observations needed for each token ID to collect
before doing the clustering

N_FREQUENT:
The most N_FREQUENT frequent token IDs will be collected to do clustering

N_CLUSTER:
The number of clusters that we want to cluster token reps into.
"""

MIN_SAMPLE_SIZE_DEV=30
MIN_SAMPLE_SIZE_FULL=1000

N_FREQUENT_DEV=10
N_FREQUENT_FULL=1000

N_CLUSTER_DEV=3
N_CLUSTER_FULL=10000

def MostFreqToken(field_src, N, min_sample_size):
    """
    It will return the ids of the N most frequent sub-words.
    ----------------
    Parameters:

    field_src: torchtext.legacy.data.Field
        This is the vocab class by Pytorch
    N: int
        The ids of the N most frequent sub-words will be returned
    min_sample_size: int
        The minimum required frequency. The least frequent tokens of N should be larger than or equal to min_sample_size
    """
    sorted_vocab={k: v for k, v in sorted(field_src.vocab.freqs.items(), key=lambda item: item[1],reverse=True)}
    frequent_vocab=[]
    frequent_ids=[]
    for count, (this_vocab, freq) in enumerate(sorted_vocab.items()):
        if count < N:
            frequent_vocab.append(this_vocab)
        else:
            break
        assert freq >= min_sample_size, "The number of frequency should be larger than the minimum required sample size"
    for this_vocab in frequent_vocab:
        frequent_ids.append(field_src.vocab.stoi[this_vocab])
    return frequent_ids


def GetInter(lst1, lst2):
    """
    Return the index of the intersected tokens between lst1 and lst2. Please be aware that the same token_id may
    show up more than once in a sentence.
    ----------------
    Parameters:

    lst1: a list of list that includes IDs in a sentence.
    lst2: The vocabulary list
    -----------------
    Returns:
    result_dict: dict
        lst3[pos]=token_id. Pos is the position of the shared token_id in lst1.
    """
    flat_list = [item for sublist in lst1 for item in sublist]
    result_dict = {}
    for idx, token_id in enumerate(flat_list):
        if token_id in lst2:
            result_dict[idx]=token_id
    return result_dict


def GetMI(X, Y, N_X, N_Y):
    """
    Return the mutual information between discrete variables X and Y

    Ref:
    https://en.wikipedia.org/wiki/Mutual_information
    https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy
    -----------------------------
    Parameters:
    X: list
        One of the two discrete variables to calculate MI
    Y: list
        One of the two discrete variables to calculate MI
    N_X: int
        The number of different classes in X
    N_Y: int
        The number of different classes in Y
    """
    c_xy = np.histogram2d(X, Y, [N_X, N_Y])[0]
    mi=mutual_info_score(None, None, contingency=c_xy)
    return mi
