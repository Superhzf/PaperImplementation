import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import dataset
from models import generate_square_subsequent_mask

import copy
import time
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# bptt is actually the intended sequence length
bptt = 100


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

def train_epoch(model: nn.Module, train_data, criterion, ntokens, optimizer,epoch, sync_every_steps) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    num_batches = len(train_data) // bptt
    optimizer.zero_grad()
    for batch, i in enumerate(range(0, len(train_data) - 1, bptt)):
        start_time = time.time()
        data, targets = get_batch(train_data, i)
        seq_len = data.size(0)
        if seq_len != bptt:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)
        curr_loss=loss.item()
        loss = loss/sync_every_steps
        loss.backward()

        if (batch+1)%sync_every_steps == 0:
            optimizer.step_and_update_lr()
            optimizer.zero_grad()

        total_loss += curr_loss
        s_per_batch = (time.time() - start_time)
        curr_ppl = math.exp(curr_loss)
        print(f'| epoch {epoch:3d} | {batch:5d}th batch | '
              f's/batch {s_per_batch:5.2f} | '
              f'loss {curr_loss:5.2f} | ppl {curr_ppl:8.2f}')
        start_time = time.time()
    return total_loss/len(train_data)

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
    return total_loss / len(eval_data)
