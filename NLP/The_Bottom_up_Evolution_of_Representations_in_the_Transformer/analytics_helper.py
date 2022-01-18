def MostFreqToken(field_src, N):
    """
    It will return the ids of the N most frequent sub-words.
    ----------------
    field_src: torchtext.legacy.data.Field
        This is the vocab class by Pytorch
    N: int
        The ids of the N most frequent sub-words will be returned
    """
    sorted_vocab={k: v for k, v in sorted(field_src.vocab.freqs.items(), key=lambda item: item[1],reverse=True)}
    frequent_vocab=[]
    frequent_ids=[]
    for count, (this_vocab, freq) in enumerate(sorted_vocab.items()):
        if count < N:
            frequent_vocab.append(this_vocab)
        else:
            break
    for this_vocab in frequent_vocab:
        frequent_ids.append(field_src.vocab.stoi[this_vocab])
    return frequent_ids
