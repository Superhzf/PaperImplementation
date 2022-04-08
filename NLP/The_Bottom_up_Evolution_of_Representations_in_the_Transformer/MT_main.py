# Modified from
# https://pytorch.org/tutorials/beginner/translation_transformer.html
# https://github.com/jadore801120/attention-is-all-you-need-pytorch
from timeit import default_timer as timer
from MT_helpers import train_epoch,evaluate
import pickle
from torchtext.legacy.data import Dataset,BucketIterator
import torch
from data_preprocessing import DEVELOPMENT_MODE
from data_preprocessing import SAVE_VOCAB_SRC, SAVE_VOCAB_TRG, SAVE_DATA_MT_TRAIN, SAVE_DATA_MT_VAL
from data_preprocessing import DATA_DIR_DEV, DATA_DIR_FULL, PAD_WORD
from data_preprocessing import SAVE_MODEL_PATH, NO_BETTER_THAN_ROUND
from models import D_MODEL, FFN_HID_DIM, NLAYERS, NHEAD, BATCH_SIZE, DROPOUT
from models import EPOCHS_DEV, EPOCHS_FULL, SYNC_EVERY_BATCH_DEV, SYNC_EVERY_BATCH_FULL
from models import SepSeq2SeqTransformer, LOSS_FN, ScheduledOptim
from models import MT_NAME
import os
import math

if DEVELOPMENT_MODE:
    DATA_DIR=DATA_DIR_DEV
    EPOCHS=EPOCHS_DEV
    SYNC_EVERY_STEPS=SYNC_EVERY_BATCH_DEV
    passed_test = False
else:
    DATA_DIR=DATA_DIR_FULL
    EPOCHS=EPOCHS_FULL
    SYNC_EVERY_STEPS=SYNC_EVERY_BATCH_FULL


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
"""
Change the length of training examples to 2 for development purpose.
BATCH_SIZE is no need to change.
"""
if DEVELOPMENT_MODE:
    train = Dataset(examples=train_examples[:2], fields=fields)
    valid = Dataset(examples=train_examples[:2], fields=fields)
else:
    train = Dataset(examples=train_examples, fields=fields)
    valid = Dataset(examples=valid_examples, fields=fields)

train_iter = BucketIterator(train, batch_size=BATCH_SIZE, device=device, train=True, shuffle=False)
valid_iter = BucketIterator(valid, batch_size=BATCH_SIZE, device=device, shuffle=False)
model=SepSeq2SeqTransformer(src_vocab_size=src_vocab_size,
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
best_model=None
# round is used to count how many epocs have been passed since no better result
# returns.
round=0

for epoch in range(1, EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(model, optimizer, loss_fn, train_iter, src_pad_idx,
                            trg_pad_idx,epoch, SYNC_EVERY_STEPS)
    end_time = timer()
    valid_loss = evaluate(model, loss_fn, train_iter,trg_pad_idx,src_pad_idx)
    valid_ppl = math.exp(valid_loss)
    if valid_loss<best_loss:
        best_loss=valid_loss
        best_ppl=valid_ppl
        best_model=model
        round=0
    else:
        round+=1
        if round>=NO_BETTER_THAN_ROUND:
            torch.save(best_model, os.path.join(SAVE_MODEL_PATH,MT_NAME))
            break
    print(f"Epoch: {epoch}|Train loss: {train_loss:.2f}|"
            f"Current validation loss: {valid_loss:.2f}|"
            f"Current validation ppl: {valid_ppl:.2f}|"
            f"Best validation loss: {best_loss:.2f}|"
            f"Best validation ppl: {best_ppl:.2f}|"
            f"Epoch time = {(end_time - start_time):.2f}s")
    if DEVELOPMENT_MODE and math.isclose(valid_loss,0,rel_tol=0.0001,abs_tol=0.0001):
        passed_test = True
        break

if not DEVELOPMENT_MODE:
    print(f"Training is done! The best model has been save to {os.path.join(SAVE_MODEL_PATH,MT_NAME)}")
else:
    if passed_test:
        print("Congrats! Your model has passed the pre-test!")
    else:
        assert 1==0, "Your model cannot overfit one batch with two observations, check it out!"
