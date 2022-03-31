from torch import Tensor
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from models import generate_square_subsequent_mask
import time
from data_preprocessing import BOS_WORD, EOS_WORD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model: nn.Module, train_iter, criterion, ntokens, optimizer,
                epoch, sync_every_steps,src_pad_idx,eos_idx):
    model.train()
    losses=0
    i=1
    optimizer.zero_grad()
    for batch in train_iter:
        start_time = time.time()
        src_seq = batch.src
        """
        Per the paper author,
        If the tokens are "<BOS> it is nice to meet you <EOS>"

        Input author:  "it is nice to meet you <EOS>"
        Output author: "is nice to meet you <EOS> <EOS>"

        It is kind of weird to me because per the PyTorch tutorial
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        They only shift the tokens to the right without adding another <EOS>
        at the end.

        Anyway, at this time, I follow the authors strategy.

        Ref: how to insert before a element in Pytorch tensor
        https://stackoverflow.com/questions/65932919/pytorch-how-to-insert-before-a-certain-element
        """
        input = src_seq[1:].to(device)
        # find out the index of eos_idx
        eos_pos = (src_seq == eos_idx).int().argmax(axis=0)
        # append another eos_idx after eos_idx
        src_seq = torch.stack([torch.cat([xi[:bpi], torch.tensor([eos_idx]), xi[bpi:]]) \
                        for xi, bpi in zip(src_seq.unbind(1), eos_pos)],dim=1)
        target = src_seq[2:].view(-1).to(device)
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
        input = src_seq[1:].to(device)
        eos_pos = (src_seq == eos_idx).int().argmax(axis=0)
        src_seq = torch.stack([torch.cat([xi[:bpi], torch.tensor([eos_idx]), xi[bpi:]]) for xi, bpi in zip(src_seq.unbind(1), eos_pos)],dim=1)
        target = src_seq[1:].view(-1).to(device)
        seq_len=input.size(0)
        src_mask=generate_square_subsequent_mask(seq_len).to(device)
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)
        losses += loss.item()
        print(f"Round {i} in the validation stage")
        i+=1

    return losses / len(val_iter)
