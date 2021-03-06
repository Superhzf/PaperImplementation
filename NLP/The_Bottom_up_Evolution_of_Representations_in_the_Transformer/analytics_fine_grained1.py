"""
Notes:
This script generates data for the pictures in part 5.3.1 figure 4
"""

import os
import torch
from data_preprocessing import DATA_DIR_DEV, SAVE_DATA_MT_TRAIN
from data_preprocessing import SAVE_VOCAB_SRC, SAVE_VOCAB_TRG, PAD_WORD
import pickle
from torchtext.legacy.data import Dataset, BucketIterator
from data_preprocessing import SAVE_MODEL_PATH, DEVELOPMENT_MODE
from MT_helpers import patch_trg, create_mask
from models import TransformerModel, Seq2SeqTransformer, generate_square_subsequent_mask
from models import LM_NAME, MLM_NAME, MT_NAME, NUM2WORD, NLAYERS
import numpy as np
from analytics_helper import GetIntermediateValues, MIN_SAMPLE_SIZE_FULL, MIN_SAMPLE_SIZE_DEV
from analytics_helper import NFreqToken, UPPERBOUND_LIST_DEV, LOWERBOUND_LIST_DEV
from analytics_helper import UPPERBOUND_LIST_FULL, LOWERBOUND_LIST_FULL
from analytics_helper import GetIntersect, GetInterValuesCCA2
from svcca.pwcca import compute_pwcca

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if DEVELOPMENT_MODE:
    data_dir=DATA_DIR_DEV
    lower_bound_list=LOWERBOUND_LIST_DEV
    upper_bound_list=UPPERBOUND_LIST_DEV
    assert len(lower_bound_list)==len(upper_bound_list)
    min_sample_size=MIN_SAMPLE_SIZE_DEV
else:
    data_dir=DATA_DIR_FULL
    lower_bound_list=LOWERBOUND_LIST_FULL
    upper_bound_list=UPPERBOUND_LIST_FULL
    assert len(lower_bound_list)==len(upper_bound_list)
    min_sample_size=MIN_SAMPLE_SIZE_FULL

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
frequent_vocab_list=[]
for this_lower_bound, this_upper_bound in zip(lower_bound_list,upper_bound_list):
    frequent_vocab = NFreqToken(field_src, this_lower_bound, this_upper_bound)
    frequent_vocab_list.append(frequent_vocab)
    print(f"{len(frequent_vocab)} tokens lie in the range")



MODELS=[LM_NAME, MLM_NAME, MT_NAME]

for this_vocab_list in frequent_vocab_list:
    """
    the structure of token_reps_model is (assuming we need 2 tokens, and 3 layers):
    {
        model_name1:[{token_id1:[],token_id2:[]},
                     {token_id1:[],token_id2:[]},
                     {token_id1:[],token_id2:[]}],
        model_name2:[{token_id1:[],token_id2:[]},
                     {token_id1:[],token_id2:[]},
                     {token_id1:[],token_id2:[]}],
        model_name3:[{token_id1:[],token_id2:[]},
                     {token_id1:[],token_id2:[]},
                     {token_id1:[],token_id2:[]}],
    }
    """
    token_reps_model={}
    for this_model_name in MODELS:
        token_reps_list=[]
        for _ in range(NLAYERS):
            this_token_reps={}
            for this_token_id in this_vocab_list:
                this_token_reps[this_token_id]=[]
            token_reps_list.append(this_token_reps)
        if this_model_name.startswith("MLM"):
            token_reps_model[f"{MLM_NAME.split('.')[0]}_SAME"]=token_reps_list
        elif this_model_name.startswith("LM"):
            token_reps_model[this_model_name.split('.')[0]]=token_reps_list
        elif this_model_name.startswith("MT"):
            token_reps_model[this_model_name.split('.')[0]]=token_reps_list

    """
    Since the inputs for all 3 tasks are the same, so we only need one
    sample_size_dict to note how many tokens we have collected.
    """
    sample_size_dict={}
    for this_token_id in this_vocab_list:
        sample_size_dict[this_token_id]=0

    for batch in train_iter:
        this_src = batch.src[1:]
        target_sample=GetIntersect(this_src.detach().numpy(), this_vocab_list)
        if len(target_sample) <= 0:
            continue

        """
        updated_sample means whether sample_size_dict has been updated.
        sample_size_dict should only be updated once for a sentence.
        """
        updated_sample = False

        src_seq_MT = this_src.to(device)
        trg = batch.trg
        trg_seq_MT, gold = map(lambda x: x.to(device), patch_trg(trg, trg_pad_idx))
        trg_seq_MT = trg_seq_MT.to(device)

        src_seq_MLM_SAME = this_src.to(device)

        src_seq_LM = this_src

        for this_model_name in MODELS:
            this_model = torch.load(os.path.join(SAVE_MODEL_PATH,this_model_name))
            this_model.eval()
            if this_model_name.startswith("MT"):
                src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_mask(src_seq_MT, trg_seq_MT, src_pad_idx, trg_pad_idx)
                _ = this_model(src=src_seq_MT,
                               src_mask=src_mask,
                               trg=trg_seq_MT,
                               tgt_mask=trg_mask,
                               src_padding_mask=src_padding_mask,
                               tgt_padding_mask=trg_padding_mask,
                               memory_key_padding_mask=src_padding_mask)
                token_reps_list=token_reps_model[MT_NAME.split('.')[0]]
                updated_sample = GetIntermediateValues(this_model,
                                                       target_sample,
                                                       NUM2WORD,
                                                       token_reps_list,
                                                       sample_size_dict,
                                                       min_sample_size,
                                                       NLAYERS,
                                                       updated_sample)

            elif this_model_name.startswith("MLM"):
                src_mask = generate_square_subsequent_mask(src_seq_MLM_SAME.size(0))
                src_padding_mask = (src_seq_MLM_SAME == src_pad_idx).transpose(0, 1)
                _ = this_model(src_seq_MLM_SAME, src_mask.to(device),src_padding_mask.to(device))
                token_reps_list=token_reps_model[f"{MLM_NAME.split('.')[0]}_SAME"]
                updated_sample = GetIntermediateValues(this_model,
                                                       target_sample,
                                                       NUM2WORD,
                                                       token_reps_list,
                                                       sample_size_dict,
                                                       min_sample_size,
                                                       NLAYERS,
                                                       updated_sample)


            elif this_model_name.startswith("LM")>0:
                src_mask = generate_square_subsequent_mask(src_seq_LM.size(0))
                src_padding_mask = (src_seq_LM == src_pad_idx).transpose(0, 1)
                _ = this_model(src_seq_LM, src_mask.to(device),src_padding_mask.to(device))
                token_reps_list=token_reps_model[this_model_name.split('.')[0]]
                updated_sample = GetIntermediateValues(this_model,
                                                       target_sample,
                                                       NUM2WORD,
                                                       token_reps_list,
                                                       sample_size_dict,
                                                       min_sample_size,
                                                       NLAYERS,
                                                       updated_sample)

        # we only need to keep the minimum sample size that has been collected
        this_min_sample_size=float('inf')
        for token_id, size in sample_size_dict.items():
            if size<this_min_sample_size:
                this_min_sample_size=size

        is_enough=True
        if this_min_sample_size<min_sample_size:
            is_enough=False

        if is_enough:
            break


    # calculate CCA
    Matrix_MT=[]
    Matrix_LM=[]
    Matrix_MLM=[]
    for this_model_name in MODELS:
        if this_model_name.startswith("MLM"):
            token_reps_list=token_reps_model[f"{this_model_name.split('.')[0]}_SAME"]
            Matrix_MLM=GetInterValuesCCA2(token_reps_list, NLAYERS, Matrix_MLM)
        elif this_model_name.startswith("LM"):
            token_reps_list=token_reps_model[f"{this_model_name.split('.')[0]}"]
            Matrix_LM=GetInterValuesCCA2(token_reps_list, NLAYERS, Matrix_LM)
        elif this_model_name.startswith("MT"):
            token_reps_list=token_reps_model[f"{this_model_name.split('.')[0]}"]
            Matrix_MT=GetInterValuesCCA2(token_reps_list, NLAYERS, Matrix_MT)
    print("Matrix_MT.shape",Matrix_MT.shape)
    print("Matrix_MLM.shape",Matrix_MLM.shape)
    print("Matrix_LM.shape",Matrix_LM.shape)
    for i in range(NLAYERS-1):
        print(f"PWCCA between layer {i+1} and layer {i+2} of MT model", compute_pwcca(Matrix_MT[i].transpose(),Matrix_MT[i+1].transpose())[0])
        print(f"PWCCA between layer {i+1} and layer {i+2} of LM model", compute_pwcca(Matrix_LM[i].transpose(),Matrix_LM[i+1].transpose())[0])
        print(f"PWCCA between layer {i+1} and layer {i+2} of MLM model", compute_pwcca(Matrix_MLM[i].transpose(),Matrix_MLM[i+1].transpose())[0])
        print('-'*50)
