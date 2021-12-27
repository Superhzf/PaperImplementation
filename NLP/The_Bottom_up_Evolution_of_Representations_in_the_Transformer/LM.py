# https://github.com/pytorch/examples/tree/master/word_language_model
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
from model import batchify, train, export_onnx, bptt, TransformerModel, evaluate
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
batch_size = 20
eval_batch_size = 10
# In development mode, I use a small dataset for faster iteration.
corpus = Corpus(path=data_source,development_mode=DEVELOPMENT_MODE)

train_data = batchify(corpus.train, batch_size)
val_data = batchify(corpus.val, eval_batch_size)
ntokens = len(corpus.dictionary)
emsize = 200
d_hid = 200
nlayers = 2
nhead = 2
dropout = 0.2
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

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
    train(model,train_data, criterion, ntokens, optimizer)
    val_loss = evaluate(model, val_data, ntokens, criterion)
    val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
    scheduler.step()

# Export the model in ONNX format.
export_onnx(f'{models_folder}LM_model.onnx', batch_size=1, seq_len=bptt,model=best_model)
