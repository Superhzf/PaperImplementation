"""
Notes:
This script generates data for the pictures in part 5.3.1 firgure 5.
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
from analytics_helper import GetInterValuesCCA3, MIN_SAMPLE_SIZE_FULL, MIN_SAMPLE_SIZE_DEV
from analytics_helper import NFreqToken, UPPERBOUND_LIST_DEV, LOWERBOUND_LIST_DEV
from analytics_helper import UPPERBOUND_LIST_FULL, LOWERBOUND_LIST_FULL
from analytics_helper import GetInterExcept, GetInterValuesCCA2
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



MODELS=[MT_NAME,LM_NAME,MLM_NAME]

for this_vocab_list in frequent_vocab_list:

    sample_size_count_MT=0
    sample_size_count_MLM=0
    sample_size_count_LM=0

    MT_matrix_mask_in=np.array([])
    MT_matrix_mask_out=np.array([])
    LM_matrix_mask_in=np.array([])
    LM_matrix_mask_out=np.array([])
    MLM_matrix_mask_in=np.array([])
    MLM_matrix_mask_out=np.array([])

    for batch in train_iter:
        src_seq_MT = batch.src.to(device)
        target_sample_MT=GetInterExcept(src_seq_MT.detach().numpy(), this_vocab_list)

        trg = batch.trg
        trg_seq_MT, gold = map(lambda x: x.to(device), patch_trg(trg, trg_pad_idx))
        trg_seq_MT = trg_seq_MT.to(device)

        src_seq_MLM_SAME = batch.src.to(device)
        target_sample_MLM_SAME=GetInterExcept(src_seq_MLM_SAME.detach().numpy(), this_vocab_list)

        src_seq_LM = batch.src[:-1]
        target_sample_LM=GetInterExcept(src_seq_LM.detach().numpy(), this_vocab_list)

        for this_model_name in MODELS:
            this_model = torch.load(os.path.join(SAVE_MODEL_PATH,this_model_name))
            this_model.eval()
            if this_model_name.startswith("MT") and len(target_sample_MT)>0:
                if sample_size_count_MT>=min_sample_size:
                    continue
                this_MT_matrix_mask_in=[[],[],[],[],[],[]]
                this_MT_matrix_mask_out=[[],[],[],[],[],[]]
                #TODO, modify the create_mask function to make it generate different src_padding_mask
                src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_mask(src_seq_MT, trg_seq_MT, src_pad_idx, trg_pad_idx)
                _ = this_model(src=src_seq_MT,
                               src_mask=src_mask,
                               trg=trg_seq_MT,
                               tgt_mask=trg_mask,
                               src_padding_mask=src_padding_mask,
                               tgt_padding_mask=trg_padding_mask,
                               memory_key_padding_mask=src_padding_mask)

                this_MT_matrix_mask_in, num_tokens=GetInterValuesCCA3(this_model, target_sample_MT, NUM2WORD,
                                                                        this_MT_matrix_mask_in, NLAYERS)
                sample_size_count_MT+=num_tokens

                this_MT_matrix_mask_in=np.array(this_MT_matrix_mask_in)
                this_MT_matrix_mask_in=np.squeeze(this_MT_matrix_mask_in,1)

                src_padding_mask2 = (src_seq_MT == src_pad_idx).transpose(0, 1)
                for pos, token_id in target_sample_MT.items():
                    src_padding_mask2 = src_padding_mask2.logical_and((src_seq_MT == token_id).transpose(0, 1))
                src_padding_mask2=torch.BoolTensor(src_padding_mask2)

                padding_mask_list_ly1=[src_padding_mask2]
                padding_mask_list_ly2=[src_padding_mask,src_padding_mask2]
                padding_mask_list_ly3=[src_padding_mask,src_padding_mask,
                                        src_padding_mask2]
                padding_mask_list_ly4=[src_padding_mask,src_padding_mask,src_padding_mask,
                               src_padding_mask2]
                padding_mask_list_ly5=[src_padding_mask,src_padding_mask,src_padding_mask,
                               src_padding_mask,src_padding_mask2]
                padding_mask_list_ly6=[src_padding_mask,src_padding_mask,src_padding_mask,
                               src_padding_mask,src_padding_mask,src_padding_mask2]
                padding_mask_list=[padding_mask_list_ly1, padding_mask_list_ly2,
                                   padding_mask_list_ly3, padding_mask_list_ly4,
                                   padding_mask_list_ly5, padding_mask_list_ly6]
                for i in range(NLAYERS):
                    _ = this_model(src=src_seq_MT,
                                   src_mask=src_mask,
                                   trg=trg_seq_MT,
                                   tgt_mask=trg_mask,
                                   src_padding_mask=src_padding_mask,
                                   tgt_padding_mask=trg_padding_mask,
                                   memory_key_padding_mask=src_padding_mask)
                    this_MT_matrix_mask_out,_=GetInterValuesCCA3(this_model,
                                                                   target_sample_MT,
                                                                   NUM2WORD,
                                                                   this_MT_matrix_mask_out,
                                                                   NLAYERS,
                                                                   i)
                this_MT_matrix_mask_out=np.array(this_MT_matrix_mask_out)
                this_MT_matrix_mask_out=np.squeeze(this_MT_matrix_mask_out,1)
                if len(MT_matrix_mask_out)<=0:
                    assert len(MT_matrix_mask_in)<=0
                    MT_matrix_mask_out=this_MT_matrix_mask_out
                    MT_matrix_mask_in=this_MT_matrix_mask_in
                else:
                    MT_matrix_mask_out=np.concatenate((MT_matrix_mask_out,this_MT_matrix_mask_out),axis=1)
                    MT_matrix_mask_in=np.concatenate((MT_matrix_mask_out,this_MT_matrix_mask_in),axis=1)
                assert MT_matrix_mask_out.shape == MT_matrix_mask_in.shape

            elif this_model_name.startswith("MLM") and len(target_sample_MLM_SAME)>0:
                if sample_size_count_MLM>=min_sample_size:
                    continue
                this_MLM_matrix_mask_in = [[],[],[],[],[],[]]
                this_MLM_matrix_mask_out = [[],[],[],[],[],[]]
                src_mask = generate_square_subsequent_mask(src_seq_MLM_SAME.size(0))
                src_padding_mask = (src_seq_MLM_SAME == src_pad_idx).transpose(0, 1)
                _ = this_model(src_seq_MLM_SAME, src_mask.to(device),src_padding_mask.to(device))

                this_MLM_matrix_mask_in, num_tokens=GetInterValuesCCA3(this_model, target_sample_MLM_SAME, NUM2WORD,
                                                                        this_MLM_matrix_mask_in, NLAYERS)
                sample_size_count_MLM+=num_tokens

                this_MLM_matrix_mask_in=np.array(this_MLM_matrix_mask_in)
                this_MLM_matrix_mask_in=np.squeeze(this_MLM_matrix_mask_in,1)

                src_padding_mask2 = (src_seq_MLM_SAME == src_pad_idx).transpose(0, 1)
                for pos, token_id in target_sample_MLM_SAME.items():
                    src_padding_mask2 = src_padding_mask2.logical_and((src_seq_MLM_SAME == token_id).transpose(0, 1))
                src_padding_mask2=torch.BoolTensor(src_padding_mask2)

                padding_mask_list_ly1=[src_padding_mask2]
                padding_mask_list_ly2=[src_padding_mask,src_padding_mask2]
                padding_mask_list_ly3=[src_padding_mask,src_padding_mask,
                                        src_padding_mask2]
                padding_mask_list_ly4=[src_padding_mask,src_padding_mask,src_padding_mask,
                               src_padding_mask2]
                padding_mask_list_ly5=[src_padding_mask,src_padding_mask,src_padding_mask,
                               src_padding_mask,src_padding_mask2]
                padding_mask_list_ly6=[src_padding_mask,src_padding_mask,src_padding_mask,
                               src_padding_mask,src_padding_mask,src_padding_mask2]
                padding_mask_list=[padding_mask_list_ly1, padding_mask_list_ly2,
                                   padding_mask_list_ly3, padding_mask_list_ly4,
                                   padding_mask_list_ly5, padding_mask_list_ly6]
                for i in range(NLAYERS):
                    _ = this_model(src_seq_MLM_SAME, src_mask.to(device),padding_mask_list[i])
                    this_MLM_matrix_mask_out,_=GetInterValuesCCA3(this_model,
                                                                   target_sample_MLM_SAME,
                                                                   NUM2WORD,
                                                                   this_MLM_matrix_mask_out,
                                                                   NLAYERS,
                                                                   i)
                this_MLM_matrix_mask_out=np.array(this_MLM_matrix_mask_out)
                this_MLM_matrix_mask_out=np.squeeze(this_MLM_matrix_mask_out,1)
                if len(MLM_matrix_mask_out)<=0:
                    assert len(MLM_matrix_mask_in)<=0
                    MLM_matrix_mask_out=this_MLM_matrix_mask_out
                    MLM_matrix_mask_in=this_MLM_matrix_mask_in
                else:
                    MLM_matrix_mask_out=np.concatenate((MLM_matrix_mask_out,this_MLM_matrix_mask_out),axis=1)
                    MLM_matrix_mask_in=np.concatenate((MLM_matrix_mask_out,this_MLM_matrix_mask_in),axis=1)
                assert MLM_matrix_mask_out.shape == MLM_matrix_mask_in.shape

            elif this_model_name.startswith("LM") and len(target_sample_LM)>0:
                if sample_size_count_LM>=min_sample_size:
                    continue
                this_LM_matrix_mask_in = [[],[],[],[],[],[]]
                this_LM_matrix_mask_out = [[],[],[],[],[],[]]
                src_mask = generate_square_subsequent_mask(src_seq_LM.size(0))
                src_padding_mask = (src_seq_LM == src_pad_idx).transpose(0, 1)
                _ = this_model(src_seq_LM, src_mask.to(device),src_padding_mask.to(device))

                this_LM_matrix_mask_in, num_tokens=GetInterValuesCCA3(this_model, target_sample_LM, NUM2WORD,
                                                                    this_LM_matrix_mask_in, NLAYERS)
                sample_size_count_LM+=num_tokens

                this_LM_matrix_mask_in=np.array(this_LM_matrix_mask_in)
                this_LM_matrix_mask_in=np.squeeze(this_LM_matrix_mask_in,1)

                src_padding_mask2 = (src_seq_LM == src_pad_idx).transpose(0, 1)
                for pos, token_id in target_sample_LM.items():
                    src_padding_mask2 = src_padding_mask2.logical_and((src_seq_LM == token_id).transpose(0, 1))
                src_padding_mask2=torch.BoolTensor(src_padding_mask2)

                padding_mask_list_ly1=[src_padding_mask2]
                padding_mask_list_ly2=[src_padding_mask,src_padding_mask2]
                padding_mask_list_ly3=[src_padding_mask,src_padding_mask,
                                        src_padding_mask2]
                padding_mask_list_ly4=[src_padding_mask,src_padding_mask,src_padding_mask,
                               src_padding_mask2]
                padding_mask_list_ly5=[src_padding_mask,src_padding_mask,src_padding_mask,
                               src_padding_mask,src_padding_mask2]
                padding_mask_list_ly6=[src_padding_mask,src_padding_mask,src_padding_mask,
                               src_padding_mask,src_padding_mask,src_padding_mask2]
                padding_mask_list=[padding_mask_list_ly1, padding_mask_list_ly2,
                                   padding_mask_list_ly3, padding_mask_list_ly4,
                                   padding_mask_list_ly5, padding_mask_list_ly6]
                for i in range(NLAYERS):
                    _ = this_model(src_seq_LM, src_mask.to(device),padding_mask_list[i])
                    this_LM_matrix_mask_out,_=GetInterValuesCCA3(this_model,
                                                                   target_sample_LM,
                                                                   NUM2WORD,
                                                                   this_LM_matrix_mask_out,
                                                                   NLAYERS,
                                                                   i)
                this_LM_matrix_mask_out=np.array(this_LM_matrix_mask_out)
                this_LM_matrix_mask_out=np.squeeze(this_LM_matrix_mask_out,1)
                if len(LM_matrix_mask_out)<=0:
                    assert len(LM_matrix_mask_in)<=0
                    LM_matrix_mask_out=this_LM_matrix_mask_out
                    LM_matrix_mask_in=this_LM_matrix_mask_in
                else:
                    LM_matrix_mask_out=np.concatenate((LM_matrix_mask_out,this_LM_matrix_mask_out),axis=1)
                    LM_matrix_mask_in=np.concatenate((LM_matrix_mask_out,this_LM_matrix_mask_in),axis=1)
                assert LM_matrix_mask_out.shape == LM_matrix_mask_in.shape

        if sample_size_count_LM>=min_sample_size and \
            sample_size_count_MLM>=min_sample_size and \
            sample_size_count_MT>=min_sample_size:
            break

    # calculate CCA
    print("MT_matrix_mask_in.shape",MT_matrix_mask_in.shape)
    print("MT_matrix_mask_out.shaoe",MT_matrix_mask_out.shape)

    print("MLM_matrix_mask_in.shape",MLM_matrix_mask_in.shape)
    print("MLM_matrix_mask_out.shaoe",MLM_matrix_mask_out.shape)

    print("LM_matrix_mask_in.shape",LM_matrix_mask_in.shape)
    print("LM_matrix_mask_out.shaoe",LM_matrix_mask_out.shape)
    for this_model_name in MODELS:
        for this_layer in range(NLAYERS):
            if this_model_name.startswith("MLM"):
                this_value=compute_pwcca(MLM_matrix_mask_in[this_layer].transpose(),
                                MLM_matrix_mask_out[this_layer].transpose())[0]
                print(f"PWCCA of model MLM in layer {this_layer+1} is {this_value}")
            elif this_model_name.startswith("LM"):
                this_value=compute_pwcca(LM_matrix_mask_in[this_layer].transpose(),
                                LM_matrix_mask_out[this_layer].transpose())[0]
                print(f"PWCCA of model LM in layer {this_layer+1} is {this_value}")
            elif this_model_name.startswith("MT"):
                this_value=compute_pwcca(MT_matrix_mask_in[this_layer].transpose(),
                                MT_matrix_mask_out[this_layer].transpose())[0]
                print(f"PWCCA of model MT in layer {this_layer+1} is {this_value}")
