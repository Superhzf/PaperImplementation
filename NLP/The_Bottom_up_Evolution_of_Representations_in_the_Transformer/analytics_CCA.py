import os
import torch
from data_preprocessing import DATA_DIR_DEV, SAVE_DATA_MT_TRAIN
from data_preprocessing import SAVE_VOCAB_SRC, SAVE_VOCAB_TRG, PAD_WORD
import pickle
from torchtext.legacy.data import Dataset, BucketIterator
from data_preprocessing import SAVE_MODEL_PATH, DEVELOPMENT_MODE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if DEVELOPMENT_MODE:
    data_dir=DATA_DIR_DEV

else:
    data_dir=DATA_DIR_FULL

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
