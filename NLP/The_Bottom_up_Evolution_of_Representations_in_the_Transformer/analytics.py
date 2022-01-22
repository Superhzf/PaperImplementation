from sklearn.cluster import MiniBatchKMeans
import numpy as np

import torch
from models import TransformerModel, Seq2SeqTransformer, generate_square_subsequent_mask
from models import LM_NAME, MLM_NAME, MT_NAME, NLAYERS, NUM2WORD
import os
from data_preprocessing import DATA_DIR_DEV, SAVE_DATA_MT_TRAIN
from data_preprocessing import SAVE_VOCAB_SRC, SAVE_VOCAB_TRG, PAD_WORD
import pickle
from torchtext.legacy.data import Dataset, BucketIterator
import pandas as pd
from analytics_helper import MostFreqToken, GetInter, GetMI, GetInterValues
from analytics_helper import MIN_SAMPLE_SIZE_DEV, MIN_SAMPLE_SIZE_FULL
from analytics_helper import N_FREQUENT_DEV, N_FREQUENT_FULL
from analytics_helper import N_CLUSTER_DEV, N_CLUSTER_FULL
from data_preprocessing import SAVE_MODEL_PATH, DEVELOPMENT_MODE
from MT_helpers import patch_trg, create_mask


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if DEVELOPMENT_MODE:
    min_sample_size=MIN_SAMPLE_SIZE_DEV
    N_frequent=N_FREQUENT_DEV
    N_cluster=N_CLUSTER_DEV
    data_dir=DATA_DIR_DEV

else:
    min_sample_size=MIN_SAMPLE_SIZE_FULL
    N_frequent=N_FREQUENT_FULL
    N_cluster=N_CLUSTER_FULL
    data_dir=DATA_DIR_FULL


MI_results_INP={LM_NAME.split('.')[0]:[],
         f"{MLM_NAME.split('.')[0]}_SAME":[],
         f"{MLM_NAME.split('.')[0]}_DIFF":[],
         MT_NAME.split('.')[0]:[]}

MI_results_OUT={LM_NAME.split('.')[0]:[],
         MLM_NAME.split('.')[0]:[]}

MODELS_INP=[LM_NAME, MLM_NAME, MT_NAME]
MODELS_OUT=[LM_NAME, MLM_NAME]

vocab_pkl_src = os.path.join(data_dir, SAVE_VOCAB_SRC)
vocab_pkl_trg = os.path.join(data_dir, SAVE_VOCAB_TRG)
train_pkl = os.path.join(data_dir, SAVE_DATA_MT_TRAIN)
field_src = pickle.load(open(vocab_pkl_src, 'rb'))
field_trg = pickle.load(open(vocab_pkl_trg, 'rb'))
src_pad_idx = field_src.vocab.stoi[PAD_WORD]
trg_pad_idx = field_trg.vocab.stoi[PAD_WORD]
train_examples = pickle.load(open(train_pkl, 'rb'))
fields = {'src':field_src , 'trg':field_trg}
train = Dataset(examples=train_examples, fields=fields)
train_iter = BucketIterator(train, batch_size=1, device=device, train=True, shuffle=False)
frequent_vocab = MostFreqToken(field_src, N_frequent, min_sample_size)

# token_reps_list saves NLAYERS dicts, for ith dict, the key is the token ID,
# the value is the representation of the ID in the ith layer.
token_reps_model={}
for this_model_name in MODELS_INP:
    token_reps_list=[]
    for _ in range(NLAYERS):
        this_token_reps={}
        for this_token_id in frequent_vocab:
            this_token_reps[this_token_id]=[]
        token_reps_list.append(this_token_reps)
    if this_model_name.startswith("MLM"):
        token_reps_model[f"{MLM_NAME.split('.')[0]}_SAME"]=token_reps_list
        token_reps_model[f"{MLM_NAME.split('.')[0]}_DIFF"]=token_reps_list
    else:
        token_reps_model[this_model_name.split('.')[0]]=token_reps_list

sample_size_dict={}
for this_token_id in frequent_vocab:
    sample_size_dict[this_token_id]=0

for batch in train_iter:
    src = batch.src
    src_seq = src.to(device)
    target_sample=GetInter(src_seq.detach().numpy(), frequent_vocab)
    if len(target_sample)>0:
        for this_model_name in MODELS_INP:
            trg = batch.trg
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(trg, trg_pad_idx))
            trg_seq = trg_seq.to(device)
            this_model = torch.load(os.path.join(SAVE_MODEL_PATH,this_model_name))
            this_model.eval()
            if this_model_name.startswith("MT"):
                src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_mask(src_seq, trg_seq, src_pad_idx, trg_pad_idx)
                _ = this_model(src=src_seq,
                               src_mask=src_mask,
                               trg=trg_seq,
                               tgt_mask=trg_mask,
                               src_padding_mask=src_padding_mask,
                               tgt_padding_mask=trg_padding_mask,
                               memory_key_padding_mask=src_padding_mask)
                token_reps_list=token_reps_model[MT_NAME.split('.')[0]]
                sample_size_dict=GetInterValues(this_model, target_sample, NUM2WORD, token_reps_list, sample_size_dict, min_sample_size, NLAYERS)
            elif this_model_name.startswith("MLM"):
                src_mask = generate_square_subsequent_mask(src_seq.size(0))
                _ = this_model(src_seq, src_mask.to(device))
                token_reps_list=token_reps_model[f"{MLM_NAME.split('.')[0]}_SAME"]
                sample_size_dict=GetInterValues(this_model, target_sample, NUM2WORD, token_reps_list, sample_size_dict, min_sample_size, NLAYERS)

                input = src_seq.clone()
                src_mask = generate_square_subsequent_mask(src_seq.size(0))
                rand_value = torch.rand(src_seq.shape)
                rand_mask = (rand_value < 0.15) * (input != src_pad_idx)
                mask_idx=(rand_mask.flatten() == True).nonzero().view(-1)
                input = input.flatten()
                input[mask_idx] = 103
                input = input.view(src_seq.size())
                _ = this_model(input.to(device), src_mask.to(device))
                token_reps_list=token_reps_model[f"{MLM_NAME.split('.')[0]}_DIFF"]
                sample_size_dict=GetInterValues(this_model, target_sample, NUM2WORD, token_reps_list, sample_size_dict, min_sample_size, NLAYERS)
            elif this_model_name.startswith("LM"):
                src_seq = src_seq[:-1]
                src_mask = generate_square_subsequent_mask(src_seq.size(0))
                _ = this_model(src_seq, src_mask.to(device))
                token_reps_list=token_reps_model[LM_NAME.split('.')[0]]
                sample_size_dict=GetInterValues(this_model, target_sample, NUM2WORD, token_reps_list, sample_size_dict, min_sample_size, NLAYERS)
            else:
                assert 1==0, "The model name is not understood"

        this_min_sample_size=float('inf')
        for key, value in sample_size_dict.items():
            if value<this_min_sample_size:
                this_min_sample_size=value
    # If we have collected MIN_SAMPLE_SIZE reps for each token ID, then break
    # the length of all dicts in token_reps_list is the same, we can use the first one
    if this_min_sample_size>=min_sample_size and len(token_reps_list[0])>=N_frequent:
        break

for this_model_name in MODELS_INP:
    if this_model_name.startswith("MLM"):
        token_reps_list=token_reps_model[f"{MLM_NAME.split('.')[0]}_SAME"]
        result_list=MI_results_INP[f"{MLM_NAME.split('.')[0]}_SAME"]
        GetMI(token_reps_list, N_frequent, N_cluster, NLAYERS, result_list)

        token_reps_list=token_reps_model[f"{MLM_NAME.split('.')[0]}_DIFF"]
        result_list=MI_results_INP[f"{MLM_NAME.split('.')[0]}_DIFF"]
        GetMI(token_reps_list, N_frequent, N_cluster, NLAYERS, result_list)

    else:
        token_reps_list=token_reps_model[this_model_name.split('.')[0]]
        result_list=MI_results_INP[this_model_name.split('.')[0]]
        GetMI(token_reps_list, N_frequent, N_cluster, NLAYERS, result_list)


print("result",MI_results_INP)
