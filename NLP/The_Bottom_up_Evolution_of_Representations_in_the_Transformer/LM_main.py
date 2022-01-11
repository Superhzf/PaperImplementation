from LM_helpers import train_epoch,evaluate, batchify
import pickle
from torchtext.legacy.data import Dataset,BucketIterator
import torch
from data_preprocessing import SAVE_VOCAB_SRC, SAVE_DATA_LM_TRAIN, SAVE_DATA_LM_VAL
from data_preprocessing import DATA_DIR_DEV, DATA_DIR_FULL, DEVELOPMENT_MODE
from models import D_MODEL, FFN_HID_DIM, NLAYERS, NHEAD, BATCH_SIZE, DROPOUT
from models import EPOCHS_DEV, EPOCHS_FULL, SYNC_EVERY_BATCH_DEV, SYNC_EVERY_BATCH_FULL
from models import TransformerModel, LOSS_FN, ScheduledOptim
import os
import time
import math

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVELOPMENT_MODE:
    DATA_DIR=DATA_DIR_DEV
    EPOCHS=EPOCHS_DEV
    SYNC_EVERY_STEPS=SYNC_EVERY_BATCH_DEV
else:
    DATA_DIR=DATA_DIR_FULL
    EPOCHS=EPOCHS_FULL
    SYNC_EVERY_STEPS=SYNC_EVERY_BATCH_FULL

print("SYNC_EVERY_STEPS",SYNC_EVERY_STEPS)
vocab_pkl_src=os.path.join(DATA_DIR, SAVE_VOCAB_SRC)
train_pkl=os.path.join(DATA_DIR, SAVE_DATA_LM_TRAIN)
valid_pkl=os.path.join(DATA_DIR, SAVE_DATA_LM_VAL)

field_src=pickle.load(open(vocab_pkl_src, 'rb'))
train_examples=pickle.load(open(train_pkl, 'rb'))
valid_examples=pickle.load(open(valid_pkl, 'rb'))

fields = {'text':field_src}
train = Dataset(examples=train_examples, fields=fields)
valid = Dataset(examples=valid_examples, fields=fields)
train_iter = BucketIterator(train, batch_size=1, device=device, train=True)
train_iter = batchify(list(train_iter)[0].text, BATCH_SIZE)
valid_iter = BucketIterator(valid, batch_size=1, device=device)
valid_iter = batchify(list(valid_iter)[0].text, BATCH_SIZE)

src_vocab_size = len(field_src.vocab)
model=TransformerModel(src_vocab_size=src_vocab_size,
                       d_model=D_MODEL,
                       nhead=NHEAD,
                       dim_feedforward=FFN_HID_DIM,
                       num_encoder_layer=NLAYERS,
                       dropout=DROPOUT)
criterion = LOSS_FN()
optimizer = ScheduledOptim(model.parameters())
best_loss=float('inf')
best_ppl=None
best_model =None
# train the model
for epoch in range(1, EPOCHS + 1):
    start_time = time.time()
    train_loss=train_epoch(model,train_iter, criterion, src_vocab_size, optimizer, epoch, SYNC_EVERY_STEPS)
    end_time = time.time()
    train_ppl =math.exp(train_loss)
    valid_loss = evaluate(model, valid_iter, src_vocab_size, criterion)
    valid_ppl = math.exp(valid_loss)
    elapsed = end_time - start_time
    if valid_loss<best_loss:
        best_loss=valid_loss
        best_ppl=valid_ppl
        best_model=model
    print(f"Epoch: {epoch}|Train loss: {train_loss:.2f}|"
            f"Current validation loss: {valid_loss:.2f}|"
            f"Current validation ppl: {valid_ppl:.2f}|"
            f"Best validation loss: {best_loss:.2f}|"
            f"Best validation ppl: {best_ppl:.2f}|"
            f"Epoch time = {(end_time - start_time):.2f}s")
