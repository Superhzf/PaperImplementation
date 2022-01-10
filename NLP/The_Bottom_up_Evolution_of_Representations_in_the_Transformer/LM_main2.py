from LM_helpers2 import train_epoch,evaluate, batchify
import pickle
from torchtext.legacy.data import Dataset,BucketIterator
import torch
from data_preprocessing2 import SAVE_DATA_SRC, SAVE_DATA_LM_TRAIN
from data_preprocessing2 import DATA_DIR, PAD_WORD
from models import D_MODEL, FFN_HID_DIM, NLAYERS, NHEAD, BATCH_SIZE, DROPOUT, EPOCHS
from models import TransformerModel, LOSS_FN, ScheduledOptim
import os
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_pkl_src = os.path.join(DATA_DIR, SAVE_DATA_SRC)
train_pkl = os.path.join(DATA_DIR, SAVE_DATA_LM_TRAIN)

field_src = pickle.load(open(vocab_pkl_src, 'rb'))
train_examples = pickle.load(open(train_pkl, 'rb'))

fields = {'text':field_src}
train = Dataset(examples=train_examples, fields=fields)
train_iter = BucketIterator(train, batch_size=1, device=device, train=True)
train_iter = batchify(list(train_iter)[0].text, BATCH_SIZE)

src_vocab_size = len(field_src.vocab)
model=TransformerModel(src_vocab_size=src_vocab_size,
                       d_model=D_MODEL,
                       nhead=NHEAD,
                       dim_feedforward=FFN_HID_DIM,
                       num_encoder_layer=NLAYERS,
                       dropout=DROPOUT)
criterion = LOSS_FN()
optimizer = ScheduledOptim(model.parameters())
# train the model
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train_epoch(model,train_iter, criterion, src_vocab_size, optimizer, epoch)
    train_loss = evaluate(model, train_iter, src_vocab_size, criterion)
    train_ppl = math.exp(train_loss)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'train loss {train_loss:5.2f} | train ppl {train_ppl:8.2f}')
    print('-' * 89)
