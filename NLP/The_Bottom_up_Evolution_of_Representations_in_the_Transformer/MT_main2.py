from timeit import default_timer as timer
from MT_helpers2 import train_epoch,evaluate
import pickle
from torchtext.legacy.data import Dataset,BucketIterator
import torch
from data_preprocessing2 import DEVELOPMENT_MODE
from data_preprocessing2 import SAVE_VOCAB_SRC, SAVE_VOCAB_TRG, SAVE_DATA_MT_TRAIN, SAVE_DATA_MT_VAL
from data_preprocessing2 import DATA_DIR_DEV, DATA_DIR_FULL, PAD_WORD
from models import D_MODEL, FFN_HID_DIM, NLAYERS, NHEAD, BATCH_SIZE, DROPOUT, EPOCHS
from models import Seq2SeqTransformer, LOSS_FN, ScheduledOptim
import os
import math

if DEVELOPMENT_MODE:
    DATA_DIR=DATA_DIR_DEV
else:
    DATA_DIR=DATA_DIR_FULL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_pkl_src = os.path.join(DATA_DIR, SAVE_VOCAB_SRC)
vocab_pkl_trg = os.path.join(DATA_DIR, SAVE_VOCAB_TRG)
train_pkl = os.path.join(DATA_DIR, SAVE_DATA_MT_TRAIN)
valid_pkl = os.path.join(DATA_DIR, SAVE_DATA_MT_VAL)

field_src = pickle.load(open(vocab_pkl_src, 'rb'))
field_trg = pickle.load(open(vocab_pkl_trg, 'rb'))
train_examples = pickle.load(open(train_pkl, 'rb'))
valid_examples = pickle.load(open(valid_pkl, 'rb'))

src_pad_idx = field_src.vocab.stoi[PAD_WORD]
trg_pad_idx = field_trg.vocab.stoi[PAD_WORD]

src_vocab_size = len(field_src.vocab)
trg_vocab_size = len(field_trg.vocab)

fields = {'src':field_src , 'trg':field_trg}
train = Dataset(examples=train_examples, fields=fields)
valid = Dataset(examples=valid_examples, fields=fields)
train_iter = BucketIterator(train, batch_size=BATCH_SIZE, device=device, train=True)
valid_iter = BucketIterator(valid, batch_size=BATCH_SIZE, device=device)

model=Seq2SeqTransformer(src_vocab_size=src_vocab_size,
                         d_model=D_MODEL,
                         nhead=NHEAD,
                         dim_feedforward=FFN_HID_DIM,
                         num_encoder_layer=NLAYERS,
                         dropout=DROPOUT,
                         num_decoder_layer=NLAYERS,
                         tgt_vocab_size=trg_vocab_size)
model=model.to(device)
loss_fn=LOSS_FN(ignore_index=trg_pad_idx)
optimizer=ScheduledOptim(model.parameters())
best_loss=float('inf')
best_ppl=None
best_model =None
print("len(train_iter)",len(train_iter))
for epoch in range(1, EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(model, optimizer, BATCH_SIZE, loss_fn, train_iter, src_pad_idx, trg_pad_idx,epoch)
    end_time = timer()
    valid_loss = evaluate(model,BATCH_SIZE, loss_fn, valid_iter)
    val_ppl = math.exp(val_loss)
    if valid_loss<best_loss:
        best_loss=valid_loss
        best_ppl=val_ppl
        best_model=model
    print(f"Epoch: {epoch}|Train loss: {train_loss:.2f}|"
            f"Current validation loss: {valid_loss:.2f}|"
            f"Current validation ppl: {val_ppl:.2f}|"
            f"Best validation loss: {best_loss:.2f}|"
            f"Best validation ppl: {best_ppl:.2f}|"
            f"Epoch time = {(end_time - start_time):.2f}s")
