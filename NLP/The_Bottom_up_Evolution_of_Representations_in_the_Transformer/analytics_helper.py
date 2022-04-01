from sklearn.cluster import MiniBatchKMeans
import numpy as np
from sklearn.metrics import mutual_info_score
import pandas as pd

"""
MIN_SAMPLE_SIZE:
The minimum number of observations needed for each token ID to collect
before doing the clustering

N_FREQUENT:
The most N_FREQUENT frequent token IDs will be collected to do clustering

N_CLUSTER:
The number of clusters that we want to cluster token reps into.

UPPERBOUND_LIST and LOWERBOUND_LIST:
The range of frequency of target ids should show up in the material. The ith position
in UPPERBOUND_LIST corresponds to the ith position in LOWERBOUND_LIST.
"""

MIN_SAMPLE_SIZE_DEV=14
MIN_SAMPLE_SIZE_FULL=1000

N_FREQUENT_DEV=10
N_FREQUENT_FULL=1000

N_CLUSTER_DEV=3
N_CLUSTER_FULL=10000

MAXIMUM_SENTENCE_COUNT_DEV=50
MAXIMUM_SENTENCE_COUNT_FULL=5000

UPPERBOUND_LIST_FULL=[10,25,50,100,250,500,1000,2500,5000,10000,30000]
LOWERBOUND_LIST_FULL=[1,10,25,50,100,250,500,1000,2500,5000,10000]

UPPERBOUND_LIST_DEV=[9]
LOWERBOUND_LIST_DEV=[9]

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


def NFreqToken(field_src, lower_bound, upper_bound):
    """
    It will return the ids of tokens whose frequency is between lower_bound and
    upper_bound.
    ----------------------
    Parameters:

    field_src: torchtext.legacy.data.Field
        This is the vocab class by Pytorch
    lower_bound: int
        The lower bound of the frequency range
    upper_bound: int
        The upper bound of the frequency range

    Returns:
    frequent_ids: list
        A list of ids of tokens whose frequency is between the lower bound and
        the upper bound.
    """
    sorted_vocab={k: v for k, v in sorted(field_src.vocab.freqs.items(), key=lambda item: item[1],reverse=True)}
    frequent_vocab=[]
    frequent_ids=[]
    for this_vocab, freq in sorted_vocab.items():
        if freq < lower_bound or freq > upper_bound:
            continue
        else:
            frequent_vocab.append(this_vocab)

    for this_vocab in frequent_vocab:
        frequent_ids.append(field_src.vocab.stoi[this_vocab])
    return frequent_ids


def GetIntersect(lst1, lst2):
    """
    Return the index of the intersected tokens between lst1 and lst2.
    Please be aware that the same token_id may show up more than once in a sentence.
    ----------------
    Parameters:

    lst1: a list of list that includes IDs in a sentence.
    lst2: The vocabulary list
    -----------------
    Returns:
    result_dict: dict
        lst3[pos]=token_id. Pos is the position of the shared token_id in lst1.
    """
    flat_list1 = [item for sublist in lst1 for item in sublist]
    result_dict = {}
    for idx, token_id in enumerate(flat_list1):
        if token_id in lst2:
            result_dict[idx]=token_id
    return result_dict


def GetInterExcept(lst1, lst2):
    """
    Return the index of the tokens that are in lst1 but not in lst2 under the
    condition that lst1 and lst2 have commen tokens.
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
    result_dict={}
    have_common=False
    for idx, token_id in enumerate(flat_list):
        if token_id in lst2:
            have_common=True
            result_dict[idx]=token_id

    if have_common:
        return result_dict
    else:
        return {}


def GetMI(token_reps_list, N_frequent, N_cluster, num_layers, result_list):
    """
    Return the mutual information between input/output tokens and the intermediate
    layer values.

    Ref:
    https://en.wikipedia.org/wiki/Mutual_information
    https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy
    -----------------------------
    Parameters:
    token_reps_list: list
        token_reps_list[i]=dict saves the reps of layer i. The format is
        dict[token_id]=reps
    N_X: int
        The number of different classes in X
    N_Y: int
        The number of different classes in Y
    num_layers: int
        The number of intermediate layers
    N_cluster: int
        The number of clusters that we want to cluster reps into
    result_list: list
        Where to save the result
    """
    for i in range(num_layers):
        df=pd.DataFrame()
        this_token_reps=token_reps_list[i]
        for token_id, samples in this_token_reps.items():
            for this_sample in samples:
                this_df=pd.DataFrame(this_sample,index=[token_id])
                df=df.append(this_df)

        kmeans = MiniBatchKMeans(n_clusters=N_cluster)
        kmeans.fit(df)
        predictions=kmeans.predict(df)
        X = df.index.values
        Y = predictions.tolist()
        assert len(X)==len(Y), "the length of X and Y should be the same"
        c_xy = np.histogram2d(X, Y, [N_frequent, N_cluster])[0]
        this_MI=mutual_info_score(None, None, contingency=c_xy)
        result_list.append(this_MI)


def GetIntermediateValues(this_model, target_sample, NUM2WORD, token_reps_list, sample_size_dict, min_sample_size, num_layers,updated_sample):
    """
    The function extracts the intermediate layer values of this_model of some
    token ids. Then update this_token_resp in place and return sample_size_dict.
    -------------------------------------
    Parameters:
    this_model: Pytorch model
        This is the model from which we would extract intermediate values
    target_sample: list
        This includes the token ids that interest us
    NUM2WORD: dict
        NUM2WORD translates from layer number to layer name
    this_token_resp: dict
        It saves the reps of tokens for one of the models (ML, MLM, MT). The format
        is this_token_resp[id]=reps.
    sample_size_dict: dic
        It saves how many samples we have collected for a token id. The format
        is sample_size_dict[id]=count.
    min_sample_size: int
        The minimum required sample size needed for a token id
    num_layers: int
        The number of intermediate layers
    """
    for pos, token_id in target_sample.items():
        # For a token ID, we only collect min_sample_size reps.
        # the length of all dicts in token_reps_list is the same, the difference
        # is the layer. So, we can use the first layer.
        if sample_size_dict[token_id]<min_sample_size:
            for i in range(num_layers):
                this_token_resp=token_reps_list[i]
                this_token_resp[token_id].append(this_model.activation[f'{NUM2WORD[i+1]}_layer'][pos].detach().numpy())
            if not updated_sample:
                sample_size_dict[token_id]+=1
    return True


def GetInterValuesCCA3(this_model, target_sample, NUM2WORD, token_reps_list, num_layers,target_layer=None):
    """
    The function extracts the all intermediate layer values of this_model except
    those token ids in target_sample. Besides, it will also return
    -------------------------------------
    Parameters:
    this_model: Pytorch model
        This is the model from which we would extract intermediate values
    target_sample: list
        This includes the token ids that interest us
    NUM2WORD: dict
        NUM2WORD translates from layer number to layer name
    this_token_resp: dict
        It saves the reps of tokens for one of the models (ML, MLM, MT). The format
        is this_token_resp[id]=reps.
    num_layers: int
        The number of intermediate layers
    target_layer: int
        This parameter is used specifically for calculating the influence
        of tokens on other tokens
    """
    if target_layer is None:
        for i in range(num_layers):
            reps=this_model.activation[f'{NUM2WORD[i+1]}_layer'][:-1].detach().numpy()
            reps=np.squeeze(reps,1)
            reps=np.delete(reps,obj=list(target_sample.keys()),axis=0)
            token_reps_list[i].append(reps)
    else:
        reps=this_model.activation[f'{NUM2WORD[target_layer+1]}_layer'][:-1].detach().numpy()
        reps=np.squeeze(reps,1)
        reps=np.delete(reps,obj=list(target_sample.keys()),axis=0)
        token_reps_list[target_layer].append(reps)

    return token_reps_list, len(reps)


def GetInterValuesCCA(this_model, NUM2WORD, matrix,layer_idx):
    """
    The function extracts the intermediate layer values of this_model of all
    tokens. Be aware that we don't analyze the reps for the <EOS> token.
    """
    this_sen_rep=this_model.activation[f'{NUM2WORD[layer_idx+1]}_layer'][:-1].detach().numpy()
    this_sen_rep=np.squeeze(this_sen_rep,1)
    matrix.append(this_sen_rep)
    return matrix


def GetInterValuesCCA2(token_reps_list, nlayers, matrix):
    """
    Return the intermediate values from token_reps_list. The values are saved
    in matrix. The shape of matrix is (nlayers, n_tokens, n_neurons).
    """
    for i in range(nlayers):
        this_Matrix=np.array([])
        this_token_reps=token_reps_list[i]
        for token_id, samples in this_token_reps.items():
            samples=np.array(samples)
            samples=samples.squeeze(1)
            if len(this_Matrix) <= 0:
                this_Matrix=samples
            else:
                this_Matrix=np.concatenate((this_Matrix,samples),axis=0)
        matrix.append(this_Matrix)
    matrix=np.array(matrix)
    return matrix
