import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from models import generate_square_subsequent_mask

import copy
import time
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, dataloader, criterion, ntokens, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0
    log_interval = 200
    start_time = time.time()
    i=0
    for batch in dataloader:
        input = batch['input_ids'].permute(1, 0).clone()
        src_mask = generate_square_subsequent_mask(batch['input_ids'].size(1))
        rand_value = torch.rand(batch.input_ids.permute(1, 0).shape)
        rand_mask = (rand_value < 0.15) * (input != 101) * (input != 102) * (input != 0)
        mask_idx=(rand_mask.flatten() == True).nonzero().view(-1)
        input = input.flatten()
        input[mask_idx] = 103
        input = input.view(batch['input_ids'].permute(1, 0).size())

        out = model(input.to(device), src_mask.to(device))
        loss = criterion(out.view(-1, ntokens), batch['input_ids'].view(-1).to(device))
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        i+=1

    return total_loss/len(dataloader)

def evaluate(model: nn.Module, dataloader, ntokens: int, criterion) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    i=0
    with torch.no_grad():
        for batch in dataloader:
            input = batch['input_ids'].permute(1, 0).clone()
            src_mask = generate_square_subsequent_mask(batch['input_ids'].size(1))
            rand_value = torch.rand(batch.input_ids.permute(1, 0).shape)
            rand_mask = (rand_value < 0.15) * (input != 101) * (input != 102) * (input != 0)
            mask_idx=(rand_mask.flatten() == True).nonzero().view(-1)
            input = input.flatten()
            input[mask_idx] = 103
            input = input.view(batch['input_ids'].permute(1, 0).size())

            out = model(input.to(device), src_mask.to(device))
            loss = criterion(out.view(-1, ntokens), batch['input_ids'].view(-1).to(device))

            total_loss += loss.item()
            i+=1
    return total_loss / (len(dataloader) - 1)
