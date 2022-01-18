
def MostFreqToken(field_src, N):
    """
    It will return the N most frequent sub-words.
    ----------------
    field_src: torchtext.legacy.data.Field
        This is the vocab class by Pytorch
    N: int
        The N most frequent sub-words will be returned
    """
    sorted_vocab={k: v for k, v in sorted(field_src.vocab.freqs.items(), key=lambda item: item[1],reverse=True)}
    frequent_vocab=[]
    for count, (this_vocab, freq) in enumerate(sorted_vocab.items()):
        if count < 1000:
            frequent_vocab.append(this_vocab)
        else:
            return frequent_vocab
