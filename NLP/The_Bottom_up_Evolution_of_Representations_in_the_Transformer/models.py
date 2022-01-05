import torch
from torch import nn, Tensor
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


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
    The main model class
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
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layer)
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, src_vocab_size)
        if initialize_weights:
            self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.src_tok_emb.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor)->Tensor:
        src = self.src_tok_emb(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


class Seq2SeqTransformer(TransformerModel):
    """
    The main model class.
    self.transformer_encoder and self.transformer_decoder can be merged into
    a single torch.nn.modules.transformer class. I separate them into two
    different classes because MLM and LM tasks only use self.transformer_encoder;
    in order to make sure all three tasks share the same network architecture,
    I decide to make Seq2SeqTransformer class inherit from TransformerModel
    of MLM and LM tasks. Therefore, I have to split torch.nn.modules.transformer
    into two parts.

    ref:https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html

    TODO: Find out what *_padding_mask for from the MultiheadAttention class.
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
        decoder_layers = TransformerDecoderLayer(d_model=d_model,
                                                 nhead=nhead,
                                                 dim_feedforward=dim_feedforward,
                                                 dropout=dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layer)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.src_tok_emb.weight.data.uniform_(-initrange, initrange)
        self.tgt_tok_emb.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
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
        memory = self.transformer_encoder(src,
                                          mask=src_mask,
                                          src_key_padding_mask=src_padding_mask)
        output = self.transformer_decoder(trg,
                                          memory,
                                          tgt_mask=tgt_mask,
                                          memory_mask=None,
                                          tgt_key_padding_mask=tgt_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)
        return self.decoder(output)


def export_onnx(path, batch_size, seq_len,model):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(path)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    dummy_src = generate_square_subsequent_mask(bptt).to(device)
    torch.onnx.export(model, (dummy_input,dummy_src), path,opset_version=10)
