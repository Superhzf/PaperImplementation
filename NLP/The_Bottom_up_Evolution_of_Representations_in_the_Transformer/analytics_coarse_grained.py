"""
Notes:
This script generates data for the pictures in part 5.2.1 and 5.2.2.

The authors mention, in 5.2.1, that all tokens come from 5000
sentences; however, the entire dataset includes 5.9M sentences and they do not
mention how those 5000 sentences are selected. I will select the first 5000
sentences for now for analytics.
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
from analytics_helper import GetInterValuesCCA, MAXIMUM_SENTENCE_COUNT_DEV, MAXIMUM_SENTENCE_COUNT_FULL
from svcca.pwcca import compute_pwcca

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if DEVELOPMENT_MODE:
    data_dir=DATA_DIR_DEV
    maximum_sentence_count=MAXIMUM_SENTENCE_COUNT_DEV

else:
    data_dir=DATA_DIR_FULL
    maximum_sentence_count=MAXIMUM_SENTENCE_COUNT_FULL

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

Matrix_MT=[]
Matrix_LM=[]
Matrix_MLM=[]

MODELS=[LM_NAME, MLM_NAME, MT_NAME]

count_sentence=0
for batch in train_iter:
    this_src = batch.src[1:]
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
            if count_sentence>=1:
                this_Matrix_MT=[]
            for i in range(NLAYERS):
                if count_sentence>=1:
                    this_Matrix_MT=GetInterValuesCCA(this_model, NUM2WORD, this_Matrix_MT, i)
                else:
                    Matrix_MT=GetInterValuesCCA(this_model, NUM2WORD, Matrix_MT, i)
                    if len(Matrix_MT)>=NLAYERS:
                        Matrix_MT=np.array(Matrix_MT)

            if count_sentence>=1:
                this_Matrix_MT=np.array(this_Matrix_MT)
                Matrix_MT=np.concatenate((Matrix_MT,this_Matrix_MT),axis=1)
        elif this_model_name.startswith("MLM"):
            src_mask = generate_square_subsequent_mask(src_seq_MLM_SAME.size(0))
            src_padding_mask = (src_seq_MLM_SAME == src_pad_idx).transpose(0, 1)
            _ = this_model(src_seq_MLM_SAME, src_mask.to(device),src_padding_mask.to(device))
            if count_sentence>=1:
                this_Matrix_MLM=[]
            for i in range(NLAYERS):
                if count_sentence>=1:
                    this_Matrix_MLM=GetInterValuesCCA(this_model, NUM2WORD, this_Matrix_MLM, i)
                else:
                    Matrix_MLM=GetInterValuesCCA(this_model, NUM2WORD, Matrix_MLM, i)
                    if len(Matrix_MLM)>=NLAYERS:
                        Matrix_MLM=np.array(Matrix_MLM)

            if count_sentence>=1:
                this_Matrix_MLM=np.array(this_Matrix_MLM)
                Matrix_MLM=np.concatenate((Matrix_MLM,this_Matrix_MLM),axis=1)
        elif this_model_name.startswith("LM"):
            src_mask = generate_square_subsequent_mask(src_seq_LM.size(0))
            src_padding_mask = (src_seq_LM == src_pad_idx).transpose(0, 1)
            _ = this_model(src_seq_LM, src_mask.to(device),src_padding_mask.to(device))
            if count_sentence>=1:
                this_Matrix_LM=[]
            for i in range(NLAYERS):
                if count_sentence>=1:
                    this_Matrix_LM=GetInterValuesCCA(this_model, NUM2WORD, this_Matrix_LM, i)
                else:
                    Matrix_LM=GetInterValuesCCA(this_model, NUM2WORD, Matrix_LM, i)
                    if len(Matrix_LM)>=NLAYERS:
                        Matrix_LM=np.array(Matrix_LM)

            if count_sentence>=1:
                this_Matrix_LM=np.array(this_Matrix_LM)
                Matrix_LM=np.concatenate((Matrix_LM,this_Matrix_LM),axis=1)

    count_sentence+=1
    if count_sentence>=maximum_sentence_count:
        break

print("Matrix_MT.shape",Matrix_MT.shape)
print("Matrix_MLM.shape",Matrix_MLM.shape)
print("Matrix_LM.shape",Matrix_LM.shape)
for i in range(NLAYERS):
    print(f"PWCCA between MT and MLM in layer {i+1}", compute_pwcca(Matrix_MT[i].transpose(),Matrix_MLM[i].transpose())[0])
    print(f"PWCCA between MT and LM in layer {i+1}", compute_pwcca(Matrix_MT[i].transpose(),Matrix_LM[i].transpose())[0])
    print(f"PWCCA between LM and MLM in layer {i+1}", compute_pwcca(Matrix_LM[i].transpose(),Matrix_MLM[i].transpose())[0])
    print('-'*50)

for i in range(NLAYERS-1):
    print(f"PWCCA between layer {i+1} and layer {i+2} of MT model", compute_pwcca(Matrix_MT[i].transpose(),Matrix_MT[i+1].transpose())[0])
    print(f"PWCCA between layer {i+1} and layer {i+2} of LM model", compute_pwcca(Matrix_LM[i].transpose(),Matrix_LM[i+1].transpose())[0])
    print(f"PWCCA between layer {i+1} and layer {i+2} of MLM model", compute_pwcca(Matrix_MLM[i].transpose(),Matrix_MLM[i+1].transpose())[0])
    print('-'*50)
