import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from models import generate_square_subsequent_mask

import copy
import time
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, dataloader, criterion, ntokens, optimizer, epoch, src_pad_idx):
    model.train()
    total_loss = 0
    log_interval = 200
    start_time = time.time()
    i=0
    for batch in dataloader:
        src = batch.src
        src_seq = src.to(device)

        input = src_seq.permute(1, 0).clone()
        src_mask = generate_square_subsequent_mask(src_seq.size(1))
        rand_value = torch.rand(src_seq.permute(1, 0).shape)
        rand_mask = (rand_value < 0.15) * (input != src_pad_idx)
        mask_idx=(rand_mask.flatten() == True).nonzero().view(-1)
        input = input.flatten()
        input[mask_idx] = 103
        input = input.view(src_seq.permute(1, 0).size())

        out = model(input.to(device), src_mask.to(device))
        loss = criterion(out.view(-1, ntokens), src_seq.view(-1).to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step_and_update_lr()

        total_loss += loss.item()
        # print the information at each batch
        s_this_batch=(time.time() - start_time)
        curr_loss=loss.item()
        curr_ppl = math.exp(curr_loss)
        print(f'| epoch {epoch:3d} | {i:5d} batch | '
                  f's/batch {s_this_batch:5.2f} | '
                  f'loss {curr_loss:5.2f} | ppl {curr_ppl:8.2f}')
        i+=1

    return total_loss/len(dataloader)

def evaluate(model: nn.Module, dataloader, ntokens: int, criterion) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    i=0
    with torch.no_grad():
        for batch in dataloader:
            src = batch.src
            src_seq = src.to(device)

            input = src_seq.permute(1, 0).clone()
            src_mask = generate_square_subsequent_mask(src_seq.size(1))
            rand_value = torch.rand(src_seq.permute(1, 0).shape)
            rand_mask = (rand_value < 0.15) * (input != src_pad_idx)
            mask_idx=(rand_mask.flatten() == True).nonzero().view(-1)
            input = input.flatten()
            input[mask_idx] = 103
            input = input.view(src_seq.permute(1, 0).size())

            out = model(input.to(device), src_mask.to(device))
            loss = criterion(out.view(-1, ntokens), src_seq.view(-1).to(device))

            total_loss += loss.item()
            i+=1
    return total_loss / len(dataloader)
