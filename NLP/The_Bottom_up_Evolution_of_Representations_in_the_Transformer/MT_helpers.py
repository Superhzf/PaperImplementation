from torch import Tensor
import torch
import torch.nn as nn
# from torch.nn import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder
import math
from torch.utils.data import DataLoader
from models import generate_square_subsequent_mask

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# make sure the tokens are in order of their indices to propertly insert them into vocab
special_symbols = ['<unk>','<pad>','<bos>','<eos>']


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

        logits = model(src=src,
                       src_mask=src_mask,
                       trg=tgt_input,
                       tgt_mask=tgt_mask,
                       src_padding_mask=src_padding_mask,
                       tgt_padding_mask=tgt_padding_mask,
                       memory_key_padding_mask=src_padding_mask)
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

        logits = model(src=src,
                       src_mask=src_mask,
                       trg=tgt_input,
                       tgt_mask=tgt_mask,
                       src_padding_mask=src_padding_mask,
                       tgt_padding_mask=tgt_padding_mask,
                       memory_key_padding_mask=src_padding_mask)
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)
