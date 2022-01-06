# Modified from:
# https://www.kaggle.com/mojammel/masked-language-model-with-pytorch-transformer
# https://towardsdatascience.com/masked-language-modelling-with-bert-7d49793e5d2c
from data_preprocessing import CorpusMLM
import torch
from torch import nn
import time
import math
import copy
from torch.utils.data import DataLoader
from MLM_helpers import train, evaluate
from models import TransformerModel, export_onnx, LOSS_FN, ScheduledOptim
from models import D_MODEL, FFN_HID_DIM, NLAYERS, NHEAD, DROPOUT, EPOCHS, BATCH_SIZE
import numpy as np

def data_collate_fn_MLM(dataset_samples_list):
    arr = np.array(dataset_samples_list)
    inputs = tokenizer(text=arr.tolist(), padding='max_length', max_length=100, return_tensors='pt')
    return inputs

# In development mode, I use a small dataset for faster iteration.
DEVELOPMENT_MODE = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_source = "./data/"
models_folder = './TrainedModels/'

corpus = CorpusMLM(path=data_source,development_mode=DEVELOPMENT_MODE)
train_data = corpus.train
val_data = corpus.val
tokenizer = corpus.tokenizer
train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=data_collate_fn_MLM)
ntokens = tokenizer.vocab_size
model = TransformerModel(ntokens, D_MODEL, NHEAD, FFN_HID_DIM, NLAYERS, DROPOUT).to(device)
optimizer = ScheduledOptim(model.parameters())
criterion = LOSS_FN()

# train the model
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train_loss = train(model,train_iter, criterion, ntokens, optimizer,epoch)
    train_loss = evaluate(model, train_iter, ntokens, criterion)
    train_ppl = math.exp(train_loss)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'train loss {train_loss:5.2f} | train ppl {train_ppl:8.2f}')
    print('-' * 89)

# Export the model in ONNX format.
export_onnx(f'{models_folder}MLM_model.onnx', batch_size=BATCH_SIZE, seq_len=100,model=model)
