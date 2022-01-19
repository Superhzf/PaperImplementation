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
