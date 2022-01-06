# Modified from:
# https://github.com/pytorch/examples/tree/master/word_language_model
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
from LM_helpers import batchify, train, bptt, evaluate
from models import TransformerModel, export_onnx, ScheduledOptim, LOSS_FN
from models import D_MODEL, FFN_HID_DIM, NLAYERS, NHEAD, DROPOUT, EPOCHS, BATCH_SIZE
from data_preprocessing import Corpus
import torch
from torch import nn
import time
import math
import copy

# In development mode, I use a small dataset for faster iteration.
DEVELOPMENT_MODE = True

# Set the random seed manually for reproducibility.
torch.manual_seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_source = "./data/"
models_folder = './TrainedModels/'
# set up the the default size
corpus = Corpus(path=data_source,development_mode=DEVELOPMENT_MODE)

train_data = batchify(corpus.train, BATCH_SIZE)
ntokens = len(corpus.dictionary)
model = TransformerModel(ntokens, D_MODEL, NHEAD, FFN_HID_DIM, NLAYERS, DROPOUT).to(device)

criterion = LOSS_FN()
optimizer = ScheduledOptim(model.parameters())

# train the model
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    # train(model,train_data, criterion, ntokens, optimizer,scheduler,epoch)
    train(model,train_data, criterion, ntokens, optimizer,epoch)
    train_loss = evaluate(model, train_data, ntokens, criterion)
    train_ppl = math.exp(train_loss)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'train loss {train_loss:5.2f} | train ppl {train_ppl:8.2f}')
    print('-' * 89)

# Export the model in ONNX format.
export_onnx(f'{models_folder}LM_model.onnx', batch_size=BATCH_SIZE, seq_len=bptt,model=model)
