import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import copy
import time
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# TODO: need more information about bptt
bptt = 35

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):
    """
    This class is required because it injects some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as the embeddings so that the two can be summed.

    Specifically, RNN/LSTM inherently take the order of the input words into consideration. However, Transformers
    cannot do it naturally because in order to speed up the training procedure, it ditched the recurrence mechanism.
    As a result, each word in a sentence simultaneously flows through the Transformer’s encoder/decoder stack,
    The model itself doesn’t have any sense of position/order for each word. PositionalEncoding is used to
    incorporate the information about the order of the words.

    Ref: https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
    -----------------------------------
    Parameters:
    d_model: int
        The same as that of in the TransformerModel class.
    dropout: float
        The same as that of in the TransformerModel class.
    max_len: int
        The length of the longest possible sentence.
    """
    def __init__(self, d_model:int, dropout:float=0.1, max_len:int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # unsqueeze add one dimension at index 1
        position = torch.arange(max_len).unsqueeze(1)
        # For more details of the formula, please refer to part3.5 of the Attention Is All You Need paper
        div_term = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0)/d_model))
        # The dimension is the same as that of the embedding. In a sentence, each word has a coordinate indictaing
        # its position in the sentence. The coordinate is calculated using sine and cosine functions, the length
        # equals to embedding length. It is not clear why both sine and cosine are used instead of one.
        pe = torch.zeros(max_len, 1, d_model)
        pe[:,0,0::2] = torch.sin(position*div_term)
        pe[:,0,1::2] = torch.cos(position*div_term)
        self.register_buffer("pe",pe)

    def forward(self, x: Tensor)->Tensor:
        """
        Parameters:
        x: A tensor with the dimension of (seq_len, batch_size, embed_dim)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    The main model class
    -----------------------------------
    Parameters:
    ntoken: int
        The total number of tokens in the corpus
    d_model: int
        The number of features, it is the emb_dim parameters in the MultiheadAttention class. Or, it is the size
        of embidding layer.
    nhead: int
        The number of heads in the MultiheadAttention class.
    d_hid: int
        The number of neurons of the feed forward network model.
    nlayers: int
        The number of TransformerEncoderLayers in the TransformerEncoder class.
    dropout: float
        Both the TransformerEncoderLayers and PositionalEncoding use dropout
    """
    def __init__(self, ntoken:int, d_model:int, nhead:int, d_hid: int, nlayers:int, dropout:float=0.5):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor)->Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def train(model: nn.Module, train_data, criterion, ntokens, optimizer,scheduler,epoch) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        seq_len = data.size(0)
        if seq_len != bptt:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

def evaluate(model: nn.Module, eval_data: Tensor, ntokens: int, criterion) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            seq_len = data.size(0)
            if seq_len != bptt:
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)

def export_onnx(path, batch_size, seq_len,model):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(path)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    dummy_src = generate_square_subsequent_mask(bptt).to(device)
    torch.onnx.export(model, (dummy_input,dummy_src), path,opset_version=10)
