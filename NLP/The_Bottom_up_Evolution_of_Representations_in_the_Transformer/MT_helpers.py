from torch import Tensor
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from models import generate_square_subsequent_mask
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def patch_trg(trg, pad_idx):
    trg, gold = trg[1:, :], trg[1:, :].contiguous().view(-1)
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


def train_epoch(model, optimizer, loss_fn, train_iter, src_pad_idx, trg_pad_idx, epoch, sync_every_steps):
    model.train()
    losses=0
    i=1
    optimizer.zero_grad()
    for batch in train_iter:
        start_time = time.time()
        src = batch.src
        trg = batch.trg
        """
        If tokens are "<BOS> it is nice to meet you <EOS>"
        Per the paper author:
        Input:  "it is nice to meet you <EOS>" from the source side
        Target: "it is nice to meet you <EOS>" from the target side
        Output: "it is nice to meet you <EOS>" from the target side

        It confuses me about <BOS> or <SOS>, per the paper author, "You can add <bos> if you
        like (although, I don't think this is standard),". So, I remove <BOS> in
        the input. However, per the examples by PyTorch tutorials this
        https://pytorch.org/tutorials/beginner/translation_transformer.html
        and this
        https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
        <BOS> or <SOS> are required.

        Due to the discrepancy in <BOS>, the inputs per the paper author and per
        PyTorch tutorials are different. Per PyTorch:
        Input:  "<BOS> it is nice to meet you <EOS>" from the source side
        Target: "<BOS> it is nice to meet you" from the target side
        Output: "it is nice to meet you <EOS>" from the target side
        """
        src_seq = src[1:].to(device)
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
        curr_loss=loss.item()
        loss = loss/sync_every_steps
        loss.backward()

        if i%sync_every_steps == 0:
            optimizer.step_and_update_lr()
            optimizer.zero_grad()

        losses += curr_loss
        # print the information at each batch
        s_this_batch=(time.time() - start_time)
        curr_ppl = math.exp(curr_loss)
        print(f'| epoch {epoch:3d} | {i:5d} batch | '
                  f's/batch {s_this_batch:5.2f} | '
                  f'loss {curr_loss:5.2f} | ppl {curr_ppl:8.2f}')
        i+=1

    return losses / len(train_iter)

def evaluate(model, loss_fn, val_iter,trg_pad_idx,src_pad_idx):
    model.eval()
    losses = 0
    for batch in val_iter:
        src = batch.src
        trg = batch.trg
        src_seq = src[1:].to(device)
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

    return losses / len(val_iter)
