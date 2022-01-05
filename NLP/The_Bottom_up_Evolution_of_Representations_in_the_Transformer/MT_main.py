# credits go to
# https://pytorch.org/tutorials/beginner/translation_transformer.html
from timeit import default_timer as timer
from MT_helpers import train_epoch,evaluate, SRC_LANGUAGE, TGT_LANGUAGE
from models import Seq2SeqTransformer
from MT_helpers import special_symbols
from MT_helpers import UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX
from torchtext.vocab import build_vocab_from_iterator
import torch
from torchtext.data.utils import get_tokenizer
from typing import Iterable, List
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from data_preprocessing import CorpusMT

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def yield_tokens(data_iter: Iterable, language:str, language_index)->List[str]:
    """
    The helper function to tokenize sentences for MT model.

    data_sample[language_index[language]] returns the EN or DE sentence.
    This sentence will be tokenized by token_transform[language]
    """
    # language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))
    #TODO: what does pad_sequence do?
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

# TOOD: what does this function do?
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids:List[str]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

torch.manual_seed(0)

data_source = "./data/"
models_folder = './TrainedModels/'
# In development mode, I use a small dataset for faster iteration.
DEVELOPMENT_MODE = True
corpus = CorpusMT(data_source, development_mode=DEVELOPMENT_MODE)
train_iter = corpus.train
val_iter = corpus.val

token_transform = {}
vocab_transform = {}

# Create the source and target language tokenizer
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language = 'de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language = 'en_core_web_sm')

language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    """
    vocab_transform[0] and vocab_transform[1] are vocabularies of EN and DE.
    """
    # create vocab
    vocab_transform[ln]=build_vocab_from_iterator(yield_tokens(train_iter, ln, language_index),
                                                  min_freq=1,
                                                  specials=special_symbols,
                                                  special_first=True)

# Set UNK_IDX as the default index. This index is returned when the token is not found.
# if not set, it throws RuntimeError when the required token is not found in the vocabulary
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
NUM_EPOCHS = 18
BATCH_SIZE = 128
DROPOUT = 0.5

transformer = Seq2SeqTransformer(src_vocab_size=SRC_VOCAB_SIZE,
                                 d_model=EMB_SIZE,
                                 nhead=NHEAD,
                                 dim_feedforward=FFN_HID_DIM,
                                 num_encoder_layer=NUM_ENCODER_LAYERS,
                                 dropout=DROPOUT,
                                 num_decoder_layer=NUM_DECODER_LAYERS,
                                 tgt_vocab_size=TGT_VOCAB_SIZE)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# This is required by collate_fn
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln]=sequential_transforms(token_transform[ln],
                                             vocab_transform[ln],
                                             tensor_transform)

for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer, BATCH_SIZE, collate_fn, loss_fn, train_iter)
    end_time = timer()
    val_loss = evaluate(transformer,BATCH_SIZE,collate_fn, loss_fn, val_iter)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
