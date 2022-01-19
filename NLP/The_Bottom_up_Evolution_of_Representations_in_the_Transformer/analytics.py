from sklearn.cluster import MiniBatchKMeans
import numpy as np

import torch
from models import TransformerModel, Seq2SeqTransformer, generate_square_subsequent_mask
from models import LM_NAME, MLM_NAME, MT_NAME
import os
from data_preprocessing import DATA_DIR_DEV, SAVE_DATA_MT_TRAIN
from data_preprocessing import SAVE_VOCAB_SRC, SAVE_VOCAB_TRG
import pickle
from torchtext.legacy.data import Dataset, BucketIterator
import pandas as pd
from analytics_helper import MostFreqToken, GetInter, GetMI
from analytics_helper import MIN_SAMPLE_SIZE_DEV, MIN_SAMPLE_SIZE_FULL
from analytics_helper import N_FREQUENT_DEV, N_FREQUENT_FULL
from analytics_helper import N_CLUSTER_DEV, N_CLUSTER_FULL
from data_preprocessing import SAVE_MODEL_PATH, DEVELOPMENT_MODE


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


vocab_pkl_src = os.path.join(data_dir, SAVE_VOCAB_SRC)
vocab_pkl_trg = os.path.join(data_dir, SAVE_VOCAB_TRG)
train_pkl = os.path.join(data_dir, SAVE_DATA_MT_TRAIN)
field_src = pickle.load(open(vocab_pkl_src, 'rb'))
field_trg = pickle.load(open(vocab_pkl_trg, 'rb'))
train_examples = pickle.load(open(train_pkl, 'rb'))
fields = {'src':field_src , 'trg':field_trg}
train = Dataset(examples=train_examples, fields=fields)
train_iter = BucketIterator(train, batch_size=1, device=device, train=True, shuffle=False)
frequent_vocab = MostFreqToken(field_src, N_frequent, min_sample_size)

LM_model = torch.load(os.path.join(SAVE_MODEL_PATH,LM_NAME))
LM_model.eval()

token_reps={}
sample_size_dict={}
for this_token_id in frequent_vocab:
    token_reps[this_token_id]=[]
    sample_size_dict[this_token_id]=0
for batch in train_iter:
    src = batch.src
    src_seq = src.to(device)
    target_sample=GetInter(src_seq.detach().numpy(), frequent_vocab)
    if len(target_sample)>0:
        src_mask = generate_square_subsequent_mask(src_seq.size(0))
        out = LM_model(src_seq, src_mask.to(device))
        for pos, token_id in target_sample.items():
            # For a token ID, we only collect min_sample_size reps.
            if len(token_reps[token_id])<min_sample_size:
                token_reps[token_id].append(LM_model.activation['first_layer'][pos].detach().numpy())
                sample_size_dict[token_id]+=1

        this_min_sample_size=float('inf')
        for key, value in sample_size_dict.items():
            if value<this_min_sample_size:
                this_min_sample_size=value
    # If we have collected MIN_SAMPLE_SIZE reps for each token ID, then break
    if this_min_sample_size>=min_sample_size and len(token_reps)>=N_frequent:
        break

df=pd.DataFrame()
for token_id, samples in token_reps.items():
    for this_sample in samples:
        this_df=pd.DataFrame(this_sample,index=[token_id])
        df=df.append(this_df)

kmeans = MiniBatchKMeans(n_clusters=N_cluster)
kmeans.fit(df)
predictions=kmeans.predict(df)
X = df.index.values
Y = predictions.tolist()
assert len(X)==len(Y), "the length of X and Y should be the same"
result=GetMI(X,Y,N_frequent,N_cluster)
print(f"The MI between the input and the first layer is {result}")
