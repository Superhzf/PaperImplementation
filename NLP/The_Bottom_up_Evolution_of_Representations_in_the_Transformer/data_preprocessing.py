"""
Modified from
https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/preprocess.py

Please be aware that the original file some bugs and it cannot run.

BOS/SOS:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
https://github.com/nicolas-ivanov/tf_seq2seq_chatbot/issues/15
https://datascience.stackexchange.com/questions/26947/why-do-we-need-to-add-start-s-end-s-symbols-when-using-recurrent-neural-n
https://discuss.pytorch.org/t/why-eos-token-in-encoder-input/64926
"""
import sys
import os
from tqdm import tqdm
from learn_bpe import learn_bpe
from apply_bpe import BPE
import codecs
from torchtext.legacy import data
from torchtext.legacy.datasets import TranslationDataset, LanguageModelingDataset
from itertools import chain
import pickle


"""
If DEVELOPMENT_MODE is true, the program will use a smaller dataset that
includes 5000 pairs of sentences for the development purpose. I recommend
running in the development mode when you run the program for the first time
to make sure no bug exists. You only have to modify it here once, other places
will cite the variable from here.
"""
DEVELOPMENT_MODE = True
"""
Settings for the preprocessing
"""
RAW_DIR = './data/'
PREFIX_DEV = "dev"
PREFIX_FULL = "dev"
_TRAIN_DATA_SOURCES_DEV ={"folder_name":"raw",
                          "src": "train_en_dev.txt",
                          "trg": "train_de_dev.txt"}
_TRAIN_DATA_SOURCES_FULL ={"folder_name":"raw",
                           "src": "train_en_full.txt",
                           "trg": "train_de_full.txt"}
_VAL_DATA_SOURCES = {"folder_name": "raw",
                     "src": "valid_en.txt",
                     "trg": "valid_de.txt"}
SAVE_VOCAB_SRC = "bpe_vocab_src.pkl"
SAVE_VOCAB_TRG = "bpe_vocab_trg.pkl"
SAVE_DATA_MT_TRAIN = "bpe_MT_train.pkl"
SAVE_DATA_MT_VAL = "bpe_MT_val.pkl"
"""
Settings for BPE
"""
DATA_DIR_DEV = './data/bpe_dev/'
DATA_DIR_FULL = './data/bpe_full/'
CODES = "codes.txt"
# symbols is the vocabulary size
SYMBOLS = 32000
MIN_FREQUENCY = 6
SEPARATOR = "@@"
PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
MAX_LEN = 200
"""
Settings for saving the best models.

Ref for saving torch models
https://pytorch.org/tutorials/beginner/saving_loading_models.html

NO_BETTER_THAN_ROUND means that once the model does not improve for
the next NO_BETTER_THAN_ROUND epochs, the training
procedure should stop. Per the authors, they monitor the performance on the
validation set and stop the training when the performance does not improve for
a while (no exact round is released). I choose NO_BETTER_THAN_ROUND=33 because
the parameters will not be updated every 16 epochs. I think the parameters
have to be updated at least 2 times before the training stops.
"""
SAVE_MODEL_PATH='./data/TrainedModel/'
NO_BETTER_THAN_ROUND=33

def mkdir_if_needed(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def extract(raw_dir, folder_name, src_filename, trg_filename):
    folder = os.path.join(raw_dir, folder_name)
    src_path = os.path.join(folder, src_filename)
    trg_path = os.path.join(folder, trg_filename)
    return src_path, trg_path


def get_raw_files(raw_dir, sources):
    raw_files = {}
    src_file, trg_file = extract(raw_dir, sources['folder_name'], sources["src"], sources["trg"])
    raw_files["src"]=(src_file)
    raw_files["trg"]=(trg_file)
    return raw_files


def encode_files(bpe, src_in_file, trg_in_file, data_dir, prefix):
    src_out_file = os.path.join(data_dir, f"{prefix}.src")
    trg_out_file = os.path.join(data_dir, f"{prefix}.trg")

    if os.path.isfile(src_out_file) and os.path.isfile(trg_out_file):
        sys.stderr.write(f"Encoded files found, skip the encoding process ...\n")

    encode_file(bpe, src_in_file, src_out_file)
    encode_file(bpe, trg_in_file, trg_out_file)
    return src_out_file, trg_out_file


def encode_file(bpe, in_file, out_file):
    sys.stderr.write(f"Read raw content from {in_file} and \n"\
            f"Write encoded content to {out_file}\n")

    with codecs.open(in_file, encoding='utf-8') as in_f:
        with codecs.open(out_file, 'w', encoding='utf-8') as out_f:
            for line in in_f:
                out_f.write(bpe.process_line(line))


def filter_examples_with_length(x):
        return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN


def main(DEVELOPMENT_MODE):
    if DEVELOPMENT_MODE:
        DATA_DIR = DATA_DIR_DEV
        _TRAIN_DATA_SOURCES = _TRAIN_DATA_SOURCES_DEV
        PREFIX = PREFIX_DEV
    else:
        DATA_DIR = DATA_DIR_FULL
        _TRAIN_DATA_SOURCES = _TRAIN_DATA_SOURCES_FULL
        PREFIX = PREFIX_FULL

    raw_train = get_raw_files(RAW_DIR, _TRAIN_DATA_SOURCES)
    raw_val = get_raw_files(RAW_DIR, _VAL_DATA_SOURCES)
    train_src = raw_train['src']
    train_trg = raw_train['trg']
    val_src = raw_val['src']
    val_trg = raw_val['trg']

    codes = os.path.join(DATA_DIR, CODES)
    learn_bpe([raw_train['src'], raw_train['trg']], codes, SYMBOLS, MIN_FREQUENCY, True)
    with codecs.open(codes, encoding='utf-8') as codes:
        bpe = BPE(codes, separator=SEPARATOR)
    encode_files(bpe, train_src, train_trg, DATA_DIR, PREFIX + '-train')
    encode_files(bpe, val_src, val_trg, DATA_DIR, PREFIX + '-val')
    field_src = data.Field(
            tokenize=str.split,
            lower=True,
            pad_token=PAD_WORD,
            init_token=BOS_WORD,
            eos_token=EOS_WORD)
    field_trg = data.Field(
            tokenize=str.split,
            lower=True,
            pad_token=PAD_WORD,
            init_token=BOS_WORD,
            eos_token=EOS_WORD)
    fields = (field_src, field_trg)

    enc_train_files_prefix = PREFIX + '-train'
    enc_val_files_prefix = PREFIX + '-val'
    train_MT = TranslationDataset(
        fields=fields,
        path=os.path.join(DATA_DIR, enc_train_files_prefix),
        exts=('.src','.trg'),
        filter_pred=filter_examples_with_length)
    valid_MT = TranslationDataset(
        fields=fields,
        path=os.path.join(DATA_DIR, enc_val_files_prefix),
        exts=('.src','.trg'),
        filter_pred=filter_examples_with_length)

    field_src.build_vocab(train_MT.src, min_freq=2)
    field_trg.build_vocab(train_MT.trg, min_freq=2)

    save_data_src = os.path.join(DATA_DIR, SAVE_VOCAB_SRC)
    save_data_trg = os.path.join(DATA_DIR, SAVE_VOCAB_TRG)
    save_data_MT_train = os.path.join(DATA_DIR, SAVE_DATA_MT_TRAIN)
    save_data_MT_valid = os.path.join(DATA_DIR, SAVE_DATA_MT_VAL)

    pickle.dump(field_src, open(save_data_src, 'wb'))
    pickle.dump(field_trg, open(save_data_trg, 'wb'))
    pickle.dump(train_MT.examples, open(save_data_MT_train, 'wb'))
    pickle.dump(valid_MT.examples, open(save_data_MT_valid, 'wb'))

if __name__ == '__main__':
    mkdir_if_needed(DATA_DIR_DEV)
    mkdir_if_needed(DATA_DIR_FULL)
    mkdir_if_needed(SAVE_MODEL_PATH)
    main(DEVELOPMENT_MODE)
