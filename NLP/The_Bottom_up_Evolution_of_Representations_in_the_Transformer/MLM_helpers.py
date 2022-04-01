import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from models import generate_square_subsequent_mask

import copy
import time
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, dataloader, criterion, ntokens, optimizer, epoch, src_pad_idx, sync_every_steps):
    model.train()
    total_loss = 0
    log_interval = 200
    start_time = time.time()
    i=1
    optimizer.zero_grad()
    for batch in dataloader:
        src = batch.src
        # we don't need the <BOS> token per the paper author
        src_seq = src[1:].to(device)

        input = src_seq.clone()
        src_mask = generate_square_subsequent_mask(src_seq.size(0))
        rand_value = torch.rand(src_seq.shape)
        rand_mask = (rand_value < 0.15) * (input != src_pad_idx)
        mask_idx=(rand_mask.flatten() == True).nonzero().view(-1)
        input = input.flatten()
        input[mask_idx] = 103
        input = input.view(src_seq.size())
        src_padding_mask = (input == src_pad_idx).transpose(0, 1)
        out = model(input.to(device), src_mask.to(device), src_padding_mask)
        loss = criterion(out.view(-1, ntokens), src_seq.view(-1).to(device))
        curr_loss=loss.item()
        loss = loss/sync_every_steps
        loss.backward()

        if i%sync_every_steps == 0:
            optimizer.step_and_update_lr()
            optimizer.zero_grad()

        total_loss += curr_loss
        # print the information at each batch
        s_this_batch=(time.time() - start_time)
        curr_ppl = math.exp(curr_loss)
        print(f'| epoch {epoch:3d} | {i:5d} batch | '
                  f's/batch {s_this_batch:5.2f} | '
                  f'loss {curr_loss:5.2f} | ppl {curr_ppl:8.2f}')
        i+=1

    return total_loss/len(dataloader)

def evaluate(model: nn.Module, dataloader, ntokens: int, criterion, src_pad_idx) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    i=0
    with torch.no_grad():
        for batch in dataloader:
            src = batch.src
            src_seq = src[1:].to(device)

            input = src_seq.clone()
            src_mask = generate_square_subsequent_mask(src_seq.size(0))
            rand_value = torch.rand(src_seq.shape)
            rand_mask = (rand_value < 0.15) * (input != src_pad_idx)
            mask_idx=(rand_mask.flatten() == True).nonzero().view(-1)
            input = input.flatten()
            input[mask_idx] = 103
            input = input.view(src_seq.size())

            out = model(input.to(device), src_mask.to(device))
            loss = criterion(out.view(-1, ntokens), src_seq.view(-1).to(device))

            total_loss += loss.item()
            i+=1
    return total_loss / len(dataloader)
