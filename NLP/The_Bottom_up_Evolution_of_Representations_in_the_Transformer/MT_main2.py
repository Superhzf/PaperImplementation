from timeit import default_timer as timer
from MT_helpers2 import train_epoch,evaluate
import pickle
from torchtext.legacy.data import Dataset,BucketIterator
import torch
from data_preprocessing2 import SAVE_DATA_SRC, SAVE_DATA_TRG, SAVE_DATA_TRAIN
from data_preprocessing2 import DATA_DIR, PAD_WORD
from models import D_MODEL, FFN_HID_DIM, NLAYERS, NHEAD, BATCH_SIZE, DROPOUT, EPOCHS
from models import Seq2SeqTransformer, LOSS_FN, ScheduledOptim
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_pkl_src = os.path.join(DATA_DIR, SAVE_DATA_SRC)
vocab_pkl_trg = os.path.join(DATA_DIR, SAVE_DATA_TRG)
train_pkl = os.path.join(DATA_DIR, SAVE_DATA_TRAIN)

field_src = pickle.load(open(vocab_pkl_src, 'rb'))
field_trg = pickle.load(open(vocab_pkl_trg, 'rb'))
train_examples = pickle.load(open(train_pkl, 'rb'))

src_pad_idx = field_src.vocab.stoi[PAD_WORD]
trg_pad_idx = field_trg.vocab.stoi[PAD_WORD]

src_vocab_size = len(field_src.vocab)
trg_vocab_size = len(field_trg.vocab)

fields = {'src':field_src , 'trg':field_trg}
train = Dataset(examples=train_examples, fields=fields)
train_iter = BucketIterator(train, batch_size=BATCH_SIZE, device=device, train=True)

model=Seq2SeqTransformer(src_vocab_size=src_vocab_size,
                         d_model=D_MODEL,
                         nhead=NHEAD,
                         dim_feedforward=FFN_HID_DIM,
                         num_encoder_layer=NLAYERS,
                         dropout=DROPOUT,
                         num_decoder_layer=NLAYERS,
                         tgt_vocab_size=trg_vocab_size)
model = model.to(device)
loss_fn = LOSS_FN(ignore_index=trg_pad_idx)
optimizer = ScheduledOptim(model.parameters())

for epoch in range(1, EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(model, optimizer, BATCH_SIZE, loss_fn, train_iter, src_pad_idx, trg_pad_idx)
    end_time = timer()
    train_loss = evaluate(model,BATCH_SIZE, loss_fn, train_iter)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
