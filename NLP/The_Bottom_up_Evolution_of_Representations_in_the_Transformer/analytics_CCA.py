"""
Notes:
Regarding CCA, the authors mention, in 5.2.1, that all tokens come from 5000
sentences without saying anything about the frequency of the tokens. Hence, I do
not filter the sentences and select the first 5000 sentences in the training
set.
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
    src_seq_MT = batch.src.to(device)
    trg = batch.trg
    trg_seq_MT, gold = map(lambda x: x.to(device), patch_trg(trg, trg_pad_idx))
    trg_seq_MT = trg_seq_MT.to(device)

    src_seq=batch.src.to(device)
    src_seq_MLM_DIFF = src_seq.clone()
    src_mask = generate_square_subsequent_mask(src_seq.size(0))
    rand_value = torch.rand(src_seq.shape)
    rand_mask = (rand_value < 0.15) * (input != src_pad_idx)
    mask_idx=(rand_mask.flatten() == True).nonzero().view(-1)
    src_seq_MLM_DIFF = src_seq_MLM_DIFF.flatten()
    src_seq_MLM_DIFF[mask_idx] = 103
    src_seq_MLM_DIFF = src_seq_MLM_DIFF.view(src_seq.size())

    src_seq_LM = batch.src[:-1]

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
                    this_Matrix_MT=GetInterValuesCCA(this_model, NUM2WORD, this_Matrix_MT, i, False)
                else:
                    Matrix_MT=GetInterValuesCCA(this_model, NUM2WORD, Matrix_MT, i, False)
                    if len(Matrix_MT)>=NLAYERS:
                        Matrix_MT=np.array(Matrix_MT)

            if count_sentence>=1:
                this_Matrix_MT=np.array(this_Matrix_MT)
                Matrix_MT=np.concatenate((Matrix_MT,this_Matrix_MT),axis=1)
        elif this_model_name.startswith("MLM"):
            src_mask = generate_square_subsequent_mask(src_seq_MLM_DIFF.size(0))
            _ = this_model(src_seq_MLM_DIFF, src_mask.to(device))
            if count_sentence>=1:
                this_Matrix_MLM=[]
            for i in range(NLAYERS):
                if count_sentence>=1:
                    this_Matrix_MLM=GetInterValuesCCA(this_model, NUM2WORD, this_Matrix_MLM, i, False)
                else:
                    Matrix_MLM=GetInterValuesCCA(this_model, NUM2WORD, Matrix_MLM, i, False)
                    if len(Matrix_MLM)>=NLAYERS:
                        Matrix_MLM=np.array(Matrix_MLM)

            if count_sentence>=1:
                this_Matrix_MLM=np.array(this_Matrix_MLM)
                Matrix_MLM=np.concatenate((Matrix_MLM,this_Matrix_MLM),axis=1)
        elif this_model_name.startswith("LM"):
            src_mask = generate_square_subsequent_mask(src_seq_LM.size(0))
            _ = this_model(src_seq_LM, src_mask.to(device))
            if count_sentence>=1:
                this_Matrix_LM=[]
            for i in range(NLAYERS):
                if count_sentence>=1:
                    this_Matrix_LM=GetInterValuesCCA(this_model, NUM2WORD, this_Matrix_LM, i, True)
                else:
                    Matrix_LM=GetInterValuesCCA(this_model, NUM2WORD, Matrix_LM, i, True)
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
