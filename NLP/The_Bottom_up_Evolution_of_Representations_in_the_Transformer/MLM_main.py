# Modified from
# https://towardsdatascience.com/masked-language-modelling-with-bert-7d49793e5d2c
# https://www.kaggle.com/mojammel/masked-language-model-with-pytorch-transformer
from MLM_helpers import train_epoch,evaluate
import pickle
from torchtext.legacy.data import Dataset,BucketIterator
import torch
from data_preprocessing import DEVELOPMENT_MODE
from data_preprocessing import SAVE_VOCAB_SRC, SAVE_VOCAB_TRG, SAVE_DATA_MT_TRAIN, SAVE_DATA_MT_VAL
from data_preprocessing import DATA_DIR_DEV, DATA_DIR_FULL, PAD_WORD
from data_preprocessing import SAVE_MODEL_PATH, NO_BETTER_THAN_ROUND
from models import D_MODEL, FFN_HID_DIM, NLAYERS, NHEAD, BATCH_SIZE, DROPOUT
from models import EPOCHS_DEV, EPOCHS_FULL, SYNC_EVERY_BATCH_DEV, SYNC_EVERY_BATCH_FULL
from models import SepTransformerModel, LOSS_FN, ScheduledOptim
from models import MLM_NAME
import os
import time
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVELOPMENT_MODE:
    DATA_DIR=DATA_DIR_DEV
    EPOCHS=EPOCHS_DEV
    SYNC_EVERY_STEPS=SYNC_EVERY_BATCH_DEV

else:
    DATA_DIR=DATA_DIR_FULL
    EPOCHS=EPOCHS_FULL
    SYNC_EVERY_STEPS=SYNC_EVERY_BATCH_FULL


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
train_iter = BucketIterator(train, batch_size=BATCH_SIZE, device=device, train=True, shuffle=False)
valid = Dataset(examples=valid_examples, fields=fields)
valid_iter = BucketIterator(valid, batch_size=BATCH_SIZE, device=device, shuffle=False)

model = SepTransformerModel(src_vocab_size=src_vocab_size,
                         d_model=D_MODEL,
                         nhead=NHEAD,
                         dim_feedforward=FFN_HID_DIM,
                         num_encoder_layer=NLAYERS,
                         dropout=DROPOUT)
model=model.to(device)
criterion = LOSS_FN(ignore_index=src_pad_idx)
optimizer = ScheduledOptim(model.parameters())
best_loss=float('inf')
best_ppl=None
best_model =None
# round is used to count how many epocs have been passed since no better result
# returns.
round=0
# train the model
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train_loss = train_epoch(model,train_iter, criterion, src_vocab_size, optimizer, epoch, src_pad_idx, SYNC_EVERY_STEPS)
    elapsed = time.time() - epoch_start_time
    valid_loss = evaluate(model, valid_iter, src_vocab_size, criterion, src_pad_idx)
    train_ppl = math.exp(train_loss)
    valid_ppl = math.exp(valid_loss)
    if valid_loss<best_loss:
        best_loss=valid_loss
        best_ppl=valid_ppl
        best_model=model
        round=0
    else:
        round+=1
        if round>=NO_BETTER_THAN_ROUND:
            torch.save(best_model, os.path.join(SAVE_MODEL_PATH,MLM_NAME))
        break
    print('-' * 89)
    print(f"Epoch: {epoch}|Train loss: {train_loss:.2f}|"
            f"Current validation loss: {valid_loss:.2f}|"
            f"Current validation ppl: {valid_ppl:.2f}|"
            f"Best validation loss: {best_loss:.2f}|"
            f"Best validation ppl: {best_ppl:.2f}|"
            f"Epoch time = {elapsed:.2f}s")
    print('-' * 89)

print(f"Training is done! The best model has been save to {os.path.join(SAVE_MODEL_PATH,MLM_NAME)}")
