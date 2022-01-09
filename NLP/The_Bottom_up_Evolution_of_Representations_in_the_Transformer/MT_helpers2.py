from torch import Tensor
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from models import generate_square_subsequent_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def patch_trg(trg, pad_idx):
    trg, gold = trg[:-1, :], trg[1:, :].contiguous().view(-1)
    return trg, gold


def create_mask(src, tgt, src_pad_idx, trg_pad_idx):
    """
    During training, we need a subsequent word mask that
    will prevent model to look into the future words when making predictions.
    We will also need masks to hide source and target padding tokens.
    """
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    #TODO: Why tgt_mask and src_mask are different.
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src == src_pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == trg_pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def train_epoch(model, optimizer, batch_size, loss_fn, train_iter, src_pad_idx, trg_pad_idx):
    model.train()
    losses=0

    i=0
    for batch in train_iter:
        src = batch.src
        trg = batch.trg
        src_seq = src.to(device)
        trg_seq, gold = map(lambda x: x.to(device), patch_trg(trg, trg_pad_idx))
        trg_seq = trg_seq.to(device)
        src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_mask(src_seq, trg_seq, src_pad_idx, trg_pad_idx)

        logits = model(src=src_seq,
                       src_mask=src_mask,
                       trg=trg_seq,
                       tgt_mask=trg_mask,
                       src_padding_mask=src_padding_mask,
                       tgt_padding_mask=trg_padding_mask,
                       memory_key_padding_mask=src_padding_mask)
        optimizer.zero_grad()

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), gold.reshape(-1))
        loss.backward()

        optimizer.step_and_update_lr()
        losses += loss.item()
        i+=1
        print(f"Round {i}")


    return losses / len(train_dataloader)

def evaluate(model, batch_size, loss_fn, val_iter):
    model.eval()
    losses = 0

    for batch in val_iter:
        src = batch.src
        trg = batch.trg
        src_seq = src.to(device)
        trg_seq, gold = map(lambda x: x.to(device), patch_trg(trg, trg_pad_idx))
        trg_seq = trg_seq.to(device)
        src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_mask(src_seq, trg_seq, src_pad_idx, trg_pad_idx)

        logits = model(src=src_seq,
                       src_mask=src_mask,
                       trg=trg_seq,
                       tgt_mask=trg_mask,
                       src_padding_mask=src_padding_mask,
                       tgt_padding_mask=trg_padding_mask,
                       memory_key_padding_mask=src_padding_mask)

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), gold.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)
