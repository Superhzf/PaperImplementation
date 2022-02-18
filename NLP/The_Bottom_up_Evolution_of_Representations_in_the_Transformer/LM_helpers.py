from torch import Tensor
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from models import generate_square_subsequent_mask
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model: nn.Module, train_iter, criterion, ntokens, optimizer,epoch, sync_every_steps,src_pad_idx):
    model.train()
    losses=0
    i=1
    optimizer.zero_grad()
    for batch in train_iter:
        start_time = time.time()
        src_seq = batch.src
        input = src_seq[:-1].to(device)
        target = src_seq[1:].view(-1).to(device)
        seq_len=input.size(0)
        src_mask=generate_square_subsequent_mask(seq_len).to(device)
        src_padding_mask = (input == src_pad_idx).transpose(0, 1)
        output = model(input, src_mask, src_padding_mask)
        loss = criterion(output.view(-1, ntokens), target)
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

def evaluate(model: nn.Module, eval_data: Tensor, ntokens: int, criterion):
    model.eval()
    losses = 0
    i=1
    for batch in eval_data:
        src = batch.src
        input = src_seq[:-1].to(device)
        target = src_seq[1:].view(-1).to(device)
        seq_len=input.size(0)
        src_mask=generate_square_subsequent_mask(seq_len).to(device)
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)
        losses += loss.item()
        print(f"Round {i} in the validation stage")
        i+=1

    return losses / len(val_iter)
