# Modified from
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# https://github.com/lyeoni/pretraining-for-language-understanding/blob/master/dataset_utils.py#L35
from timeit import default_timer as timer
from LM_helpers import train_epoch,evaluate
import pickle
from torchtext.legacy.data import Dataset,BucketIterator
import torch
from data_preprocessing import DEVELOPMENT_MODE
from data_preprocessing import SAVE_VOCAB_SRC, SAVE_VOCAB_TRG, SAVE_DATA_MT_TRAIN, SAVE_DATA_MT_VAL
from data_preprocessing import DATA_DIR_DEV, DATA_DIR_FULL, PAD_WORD, EOS_WORD, BOS_WORD
from data_preprocessing import SAVE_MODEL_PATH, NO_BETTER_THAN_ROUND_FULL, NO_BETTER_THAN_ROUND_DEV
from models import D_MODEL, FFN_HID_DIM, NLAYERS, NHEAD, BATCH_SIZE, DROPOUT
from models import EPOCHS_DEV, EPOCHS_FULL, SYNC_EVERY_BATCH_DEV, SYNC_EVERY_BATCH_FULL
from models import SepTransformerModel, LOSS_FN, ScheduledOptim
from models import LM_NAME
import os
import math


if DEVELOPMENT_MODE:
    DATA_DIR=DATA_DIR_DEV
    EPOCHS=EPOCHS_DEV
    SYNC_EVERY_STEPS=SYNC_EVERY_BATCH_DEV
    NO_BETTER_THAN_ROUND = NO_BETTER_THAN_ROUND_DEV
else:
    DATA_DIR=DATA_DIR_FULL
    EPOCHS=EPOCHS_FULL
    SYNC_EVERY_STEPS=SYNC_EVERY_BATCH_FULL
    NO_BETTER_THAN_ROUND = NO_BETTER_THAN_ROUND_FULL


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
eos_idx = field_src.vocab.stoi[EOS_WORD]
bos_ids = field_src.vocab.stoi[BOS_WORD]
# print("eos_idx",eos_idx)
# print("bos_ids",bos_ids)
src_vocab_size = len(field_src.vocab)

fields = {'src':field_src , 'trg':field_trg}
train = Dataset(examples=train_examples, fields=fields)
valid = Dataset(examples=valid_examples, fields=fields)
train_iter = BucketIterator(train, batch_size=BATCH_SIZE, device=device, train=True, shuffle=False)
valid_iter = BucketIterator(valid, batch_size=BATCH_SIZE, device=device, shuffle=False)
model=SepTransformerModel(src_vocab_size=src_vocab_size,
                          d_model=D_MODEL,
                          nhead=NHEAD,
                          dim_feedforward=FFN_HID_DIM,
                          num_encoder_layer=NLAYERS,
                          dropout=DROPOUT)

model=model.to(device)
loss_fn=LOSS_FN(ignore_index=src_pad_idx)
optimizer=ScheduledOptim(model.parameters())
best_loss=float('inf')
best_ppl=None
best_model =None
# round is used to count how many epocs have been passed since no better result
# returns.
round=0
for epoch in range(1, EPOCHS+1):
    start_time = timer()
    train_loss=train_epoch(model, train_iter, loss_fn, src_vocab_size,
                        optimizer, epoch, SYNC_EVERY_STEPS, src_pad_idx,eos_idx)
    end_time = timer()
    valid_loss = evaluate(model, valid_iter, src_vocab_size, loss_fn,src_pad_idx,eos_idx)
    valid_ppl = math.exp(valid_loss)
    if valid_loss<best_loss:
        best_loss=valid_loss
        best_ppl=valid_ppl
        best_model=model
        round=0
    else:
        round+=1
        if round>=NO_BETTER_THAN_ROUND:
            torch.save(best_model, os.path.join(SAVE_MODEL_PATH,LM_NAME))
            break
    print(f"Epoch: {epoch}|Train loss: {train_loss:.2f}|"
            f"Current validation loss: {valid_loss:.2f}|"
            f"Current validation ppl: {valid_ppl:.2f}|"
            f"Best validation loss: {best_loss:.2f}|"
            f"Best validation ppl: {best_ppl:.2f}|"
            f"Epoch time = {(end_time - start_time):.2f}s")

if epoch == EPOCHS:
    print("All the epochs have been executed!")
    torch.save(best_model, os.path.join(SAVE_MODEL_PATH,LM_NAME))

print(f"Training is done! The best model has been save to {os.path.join(SAVE_MODEL_PATH,LM_NAME)}")
