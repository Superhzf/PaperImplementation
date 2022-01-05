# ref:
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
from models import TransformerModel
import numpy as np

def data_collate_fn_MLM(dataset_samples_list):
    arr = np.array(dataset_samples_list)
    inputs = tokenizer(text=arr.tolist(), padding='max_length', max_length=5000, return_tensors='pt')
    return inputs

# In development mode, I use a small dataset for faster iteration.
DEVELOPMENT_MODE = True

# Set the random seed manually for reproducibility.
torch.manual_seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_source = "./data/"
models_folder = './TrainedModels/'
BATCH_SIZE = 4

corpus = CorpusMLM(path=data_source,development_mode=DEVELOPMENT_MODE)
train_data = corpus.train
val_data = corpus.val
tokenizer = corpus.tokenizer
train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=data_collate_fn_MLM)
val_iter = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=data_collate_fn_MLM)

ntokens = tokenizer.vocab_size
emsize = 200
nhid = 200
nlayers = 2
nhead = 2
dropout = 0.2
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

best_val_loss = float('inf')
epochs = 1
best_model = None

# train the model
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train_loss = train(model,train_iter, criterion, ntokens, optimizer,scheduler,epoch)
    val_loss = evaluate(model, val_iter, ntokens, criterion)
    val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'train loss {train_loss:5.2f}'
          f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
    scheduler.step()
