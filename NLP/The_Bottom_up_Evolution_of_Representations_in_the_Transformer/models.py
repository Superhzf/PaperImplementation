# Difference between attn_mask and key_padding_mask in MultiheadAttention
# https://stackoverflow.com/questions/62629644/what-the-difference-between-att-mask-and-key-padding-mask-in-multiheadattnetion
import torch
from torch import nn, Tensor
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import os
from typing import Union
"""
Per the paper, we need to set up seeds to make sure different tasks (LM, MLM, MT)
share the same initialization.
"""
seed_src_tok_emb =  1234
seed_decoder = 5678
seed_tgt_tok_emb = 4321
seed_transform_encoderlayer = 7890
seed_transform_decoderlayer = 9870
"""
Similarly, we need to set up the same parameters for
different tasks (LM, MLM, MT). The values come from the paper
Attention Is All You Need.
"""
D_MODEL = 512
FFN_HID_DIM = 2048
NLAYERS = 6
NHEAD = 8
DROPOUT = 0.1
EPOCHS_DEV = 1000
"""
EPOCHS_FULL comes from Attention Is All You Need, where the model is trained
100,000 steps (I believe one step means updating parameter once).

In this case, we have 500,000 sentences and batch size = 128, parameters will be
updated every 125 batches, so one epoch includes 500000/128/125 ~31 steps.
I will train 5000 steps which is ~160 epochs.
"""
EPOCHS_FULL = 160
BATCH_SIZE = 128
LM_NAME='LM.pt'
MLM_NAME='MLM.pt'
MT_NAME='MT.pt'
NUM2WORD={1:'first',2:'second',3:'third',4:'fourth',5:'fifth',6:'sixth'}
"""
SYNC_EVERY_STEPS=16 means that we want to accumulate the gradients every 16
batches. Why do we want this? Because if we want to set up the BATCH_SIZE=16000,
then we may not have enough memory to process it, so we set up BATCH_SIZE=1000
and accumulate the gradients for 16 batches before updating the parameters.

Per the paper, BATCH_SIZE=16000, so BATCH_SIZE*SYNC_EVERY_BATCH_FULL should be
16000.

Ref:
https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20?u=alband
"""
SYNC_EVERY_BATCH_DEV = 1
SYNC_EVERY_BATCH_FULL = 125
N_WARMUP_STEPS = 4000
BETAS = (0.9, 0.98)
EPS = 1e-9
"""
We also need to set up the same criterion, learning rate, and optimizer for
different tasks (LM, MLM, MT). Optimizer and learning rate are set up below
in the ScheduledOptim class.
"""
LOSS_FN = nn.CrossEntropyLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class GetActivation(object):
    """
    The is the class version of the nested function

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    ref:
    https://stackoverflow.com/questions/12019961/python-pickling-nested-functions
    https://stackoverflow.com/questions/68852988/how-to-get-output-from-intermediate-encoder-layers-in-pytorch-transformer

    """
    def __init__(self, name, activation):
        self.name=name
        self.activation=activation

    def __call__(self, model, input, output):
        self.activation[self.name] = output.detach()


class PositionalEncoding(nn.Module):
    """
    This class is required because it injects some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as the embeddings so that the two can be summed.

    Specifically, RNN/LSTM inherently take the order of the input words into consideration. However, Transformers
    cannot do it naturally because in order to speed up the training procedure, it ditched the recurrence mechanism.
    As a result, each word in a sentence simultaneously flows through the Transformer’s encoder/decoder stack,
    The model itself doesn’t have any sense of position/order for each word. PositionalEncoding is used to
    incorporate the information about the order of the words.

    Ref: https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
    -----------------------------------
    Parameters:
    d_model: int
        The same as that of in the TransformerModel class.
    dropout: float
        The same as that of in the TransformerModel class.
    max_len: int
        The length of the longest possible sentence.
    """
    def __init__(self, d_model:int, dropout:float=0.1, max_len:int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # unsqueeze add one dimension at index 1
        position = torch.arange(max_len).unsqueeze(1)
        # For more details of the formula, please refer to part3.5 of the Attention Is All You Need paper
        div_term = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0)/d_model))
        # The dimension is the same as that of the embedding. In a sentence, each word has a coordinate indictaing
        # its position in the sentence. The coordinate is calculated using sine and cosine functions, the length
        # equals to embedding length. It is not clear why both sine and cosine are used instead of one.
        pe = torch.zeros(max_len, 1, d_model)
        pe[:,0,0::2] = torch.sin(position*div_term)
        pe[:,0,1::2] = torch.cos(position*div_term)
        self.register_buffer("pe",pe)

    def forward(self, x: Tensor)->Tensor:
        """
        Parameters:
        x: A tensor with the dimension of (seq_len, batch_size, embed_dim)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    The main model class for LM and MLM tasks.
    -----------------------------------
    Parameters:
    src_vocab_size: int
        The total number of tokens in the corpus
    d_model: int
        The number of features, it is the emb_dim parameters in the MultiheadAttention class. Or, it is the size
        of embidding layer.
    nhead: int
        The number of heads in the MultiheadAttention class.
    dim_feedforward: int
        The number of neurons of the feed forward network model.
    num_encoder_layer: int
        The number of TransformerEncoderLayers in the TransformerEncoder class.
    dropout: float
        Both the TransformerEncoderLayers and PositionalEncoding use dropout
    """
    def __init__(self,
                 src_vocab_size:int,
                 d_model:int,
                 nhead:int,
                 dim_feedforward:int,
                 num_encoder_layer:int,
                 dropout:float=0.5,
                 initialize_weights:bool=True):
        super().__init__()
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        torch.manual_seed(seed_transform_encoderlayer)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.num_encoder_layer=num_encoder_layer
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.num_encoder_layer)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, src_vocab_size)
        self.activation={}
        for i in range(self.num_encoder_layer):
            name="{}_layer".format(NUM2WORD[i+1])
            self.transformer_encoder.layers[i].register_forward_hook(GetActivation(name, self.activation))

        if initialize_weights:
            self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        torch.manual_seed(seed_src_tok_emb)
        self.src_tok_emb.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        torch.manual_seed(seed_decoder)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor, padding_mask: Tensor)->Tensor:
        self.src_emb = self.src_tok_emb(src) * math.sqrt(self.d_model)
        self.src_pe = self.pos_encoder(self.src_emb)
        self.output_te = self.transformer_encoder(self.src_pe, src_mask, padding_mask)
        output = self.decoder(self.output_te)
        return output


class SepTransformerModel(nn.Module):
    """
    This is the separated version of TransformerModel class. Specifically, instead
    of using torch.nn.TransformerEncoder, I split it and construct the intermediate
    torch.nn.TransformerEncoderLayer separately. That way we can alter the "mask"
    parameter for different intermediate layers.
    """
    def __init__(self,
                 src_vocab_size:int,
                 d_model:int,
                 nhead:int,
                 dim_feedforward:int,
                 num_encoder_layer:int,
                 dropout:float=0.5,
                 initialize_weights:bool=True):
        super().__init__()
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.num_encoder_layer=num_encoder_layer
        self.transformer_encoder=nn.ModuleList()
        for this_layer in range(self.num_encoder_layer):
            torch.manual_seed(seed_transform_encoderlayer+this_layer)
            this_encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            self.transformer_encoder.append(this_encoder_layer)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, src_vocab_size)
        self.activation={}
        for i in range(self.num_encoder_layer):
            name="{}_layer".format(NUM2WORD[i+1])
            self.transformer_encoder[i].register_forward_hook(GetActivation(name, self.activation))
        if initialize_weights:
            self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        torch.manual_seed(seed_src_tok_emb)
        self.src_tok_emb.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        torch.manual_seed(seed_decoder)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor, padding_mask: Union[list, Tensor])->Tensor:
        self.src_emb = self.src_tok_emb(src) * math.sqrt(self.d_model)
        self.src_pe = self.pos_encoder(self.src_emb)
        if isinstance(padding_mask, list):
            # assert len(padding_mask) == self.num_encoder_layer
            # we don't need any assertion here because len(padding_mask) could be
            # smaller than self.num_encoder_layer because we don't need reps in
            # layers after tokens are masked out.
            for enc_layer, this_padding_mask in zip(self.transformer_encoder,padding_mask):
                self.src_pe = enc_layer(self.src_pe, src_mask, this_padding_mask.to(device))
        else:
            for enc_layer in self.transformer_encoder:
                self.src_pe = enc_layer(self.src_pe, src_mask, padding_mask)

        output = self.decoder(self.src_pe)
        return output


class Seq2SeqTransformer(TransformerModel):
    """
    The main model class for the MT task. I make it inherit from the
    TransformerModel class to make sure they share the same architecture regarding
    the encoder part.

    self.transformer_encoder and self.transformer_decoder can be merged into
    a single torch.nn.modules.transformer class. I separate them into two
    different classes because MLM and LM tasks only use self.transformer_encoder;
    in order to make sure all three tasks share the same network architecture,
    I decide to make Seq2SeqTransformer class inherit from TransformerModel
    of MLM and LM tasks. Therefore, I have to split torch.nn.modules.transformer
    into two parts.

    ref:https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html
    """
    def __init__(self,
                 src_vocab_size:int,
                 d_model:int,
                 nhead:int,
                 dim_feedforward:int,
                 num_encoder_layer:int,
                 dropout:float,
                 num_decoder_layer:int,
                 tgt_vocab_size:int):
        super().__init__(src_vocab_size,
                         d_model,nhead,
                         dim_feedforward,
                         num_encoder_layer,
                         dropout,
                         False)
        self.decoder=nn.Linear(d_model, tgt_vocab_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        torch.manual_seed(seed_transform_decoderlayer)
        decoder_layers = TransformerDecoderLayer(d_model=d_model,
                                                 nhead=nhead,
                                                 dim_feedforward=dim_feedforward,
                                                 dropout=dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layer)
        self.activation={}
        for i in range(self.num_encoder_layer):
            name="{}_layer".format(NUM2WORD[i+1])
            self.transformer_encoder.layers[i].register_forward_hook(GetActivation(name, self.activation))

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        torch.manual_seed(seed_src_tok_emb)
        self.src_tok_emb.weight.data.uniform_(-initrange, initrange)
        torch.manual_seed(seed_tgt_tok_emb)
        self.tgt_tok_emb.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        torch.manual_seed(seed_decoder)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,
                src: Tensor,
                src_mask:Tensor,
                trg: Tensor,
                tgt_mask:Tensor,
                src_padding_mask:Tensor,
                tgt_padding_mask:Tensor,
                memory_key_padding_mask: Tensor):
        src = self.src_tok_emb(src)*math.sqrt(self.d_model)
        trg = self.tgt_tok_emb(trg)*math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src)
        tgt_emb = self.pos_encoder(trg)
        memory = self.transformer_encoder(src_emb,
                                          mask=src_mask,
                                          src_key_padding_mask=src_padding_mask)
        output = self.transformer_decoder(tgt_emb,
                                          memory,
                                          tgt_mask=tgt_mask,
                                          memory_mask=None,
                                          tgt_key_padding_mask=tgt_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)
        return self.decoder(output)


class SepSeq2SeqTransformer(SepTransformerModel):
    """
    This is the separated version of Seq2SeqTransformer class. Specifically, instead
    of using torch.nn.TransformerEncoder, I split it and construct the intermediate
    torch.nn.TransformerEncoderLayer separately. That way we can alter the "mask"
    parameter for different intermediate layers.
    """
    def __init__(self,
                 src_vocab_size:int,
                 d_model:int,
                 nhead:int,
                 dim_feedforward:int,
                 num_encoder_layer:int,
                 dropout:float,
                 num_decoder_layer:int,
                 tgt_vocab_size:int):
        super().__init__(src_vocab_size,
                         d_model,nhead,
                         dim_feedforward,
                         num_encoder_layer,
                         dropout,
                         False)
        self.decoder=nn.Linear(d_model, tgt_vocab_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        torch.manual_seed(seed_transform_decoderlayer)
        decoder_layers = TransformerDecoderLayer(d_model=d_model,
                                                 nhead=nhead,
                                                 dim_feedforward=dim_feedforward,
                                                 dropout=dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layer)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        torch.manual_seed(seed_src_tok_emb)
        self.src_tok_emb.weight.data.uniform_(-initrange, initrange)
        torch.manual_seed(seed_tgt_tok_emb)
        self.tgt_tok_emb.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        torch.manual_seed(seed_decoder)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,
                src: Tensor,
                src_mask: Union[list, Tensor],
                trg: Tensor,
                tgt_mask:Tensor,
                src_padding_mask:Tensor,
                tgt_padding_mask:Tensor,
                memory_key_padding_mask: Tensor):
        src = self.src_tok_emb(src)*math.sqrt(self.d_model)
        trg = self.tgt_tok_emb(trg)*math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src)
        tgt_emb = self.pos_encoder(trg)

        if isinstance(src_padding_mask, list):
            # assert len(src_mask) == self.num_encoder_layer
            # the length of those two does not need to be the same as I mentioned
            # in SepTransformerModel class.
            for enc_layer, this_src_padding_mask in zip(self.transformer_encoder,src_padding_mask):
                src_emb = enc_layer(src_emb, src_mask, this_src_padding_mask)
        else:
            for enc_layer in self.transformer_encoder:
                src_emb = enc_layer(src_emb, src_mask, src_padding_mask)

        output = self.transformer_decoder(tgt_emb,
                                          src_emb,
                                          tgt_mask=tgt_mask,
                                          memory_mask=None,
                                          tgt_key_padding_mask=tgt_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)
        return self.decoder(output)


class ScheduledOptim():
    '''
    The same optimizer and learning rate for different tasks (LM, MLM, and MT).

    Modified from:
    https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py

    The idea is from the paper Attention Is All You Need.
    '''

    def __init__(self, model_parameters):
        self._optimizer = torch.optim.Adam(model_parameters, betas=BETAS, eps=EPS)
        self.lr_mul = 1
        self.d_model = D_MODEL
        self.n_warmup_steps = N_WARMUP_STEPS
        self.n_steps = 0


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
