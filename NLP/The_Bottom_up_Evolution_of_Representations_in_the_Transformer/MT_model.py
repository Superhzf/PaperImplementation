from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# make sure the tokens are in order of their indices to propertly insert them into vocab
special_symbols = ['<unk>','<pad>','<bos>','<eos>']

class PositionalEncoding(nn.Module):
    """
    Regarding the reason about why PositionalEncoding is required, please refer to that of the LM model
    https://github.com/Superhzf/PaperImplementation/blob/main/NLP/The_Bottom_up_Evolution_of_Representations_in_the_Transformer/LM_model.py#L26
    """
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, 1, emb_size))
        pos_embedding[:, 0, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 0, 1::2] = torch.cos(pos * den)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class Seq2SeqTransformer(nn.Module):
    """
    The main model class

    TODO: Find out what *_padding_mask for from the MultiheadAttention class.
    """
    def __init__(self,
                 num_encoder_layer:int,
                 num_decoder_layer:int,
                 emb_size:int,
                 nhead:int,
                 src_vocab_size:int,
                 tgt_vocab_size:int,
                 dim_feedforward:int=512,
                 dropout:float=0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer=Transformer(d_model=emb_size,
                                     nhead=nhead,
                                     num_encoder_layers=num_encoder_layer,
                                     num_decoder_layers=num_decoder_layer,
                                     dim_feedforward=dim_feedforward,
                                     dropout=dropout)
        self.decoder=nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.src_tok_emb.weight.data.uniform_(-initrange, initrange)
        self.tgt_tok_emb.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask:Tensor,
                tgt_mask:Tensor,
                src_padding_mask:Tensor,
                tgt_padding_mask:Tensor,
                memory_key_padding_mask: Tensor):
        src = self.src_tok_emb(src)
        trg = self.tgt_tok_emb(trg)
        src_emb = self.positional_encoding(src)
        tgt_emb = self.positional_encoding(trg)
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.decoder(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        # This should be the same as the TransformerEncoderLayer
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def deconder(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        # This should be the same as the TransformerDecoderLayer
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    """
    During training, we need a subsequent word mask that
    will prevent model to look into the future words when making predictions.
    We will also need masks to hide source and target padding tokens.
    """
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    #TODO: Why tgt_mask and src_mask are different.
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def train_epoch(model, optimizer, batch_size, collate_fn, loss_fn, train_iter):
    model.train()
    losses = 0
    train_dataloader = DataLoader(train_iter, batch_size=batch_size, collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        #TODO: why end at -1?
        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()
        #TOOD: why start at 1?
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)

def evaluate(model, batch_size,collate_fn, loss_fn, val_iter):
    model.eval()
    losses = 0

    val_dataloader = DataLoader(val_iter, batch_size=batch_size, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)
