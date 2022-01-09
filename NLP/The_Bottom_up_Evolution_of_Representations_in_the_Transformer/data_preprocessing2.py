"""
Modified from
https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/preprocess.py
"""
import sys
import os
from tqdm import tqdm
from learn_bpe import learn_bpe
from apply_bpe import BPE
import codecs
from torchtext.legacy import data
from torchtext.legacy.datasets import TranslationDataset
from itertools import chain
import pickle

"""
Settings for the preprocessing
"""
RAW_DIR = './data2/'
PREFIX_NC = "nc"
_TRAIN_DATA_SOURCES = [
    {
     "folder_name":"training",
     "src": "news-commentary-v12.de-en.en",
     "trg": "news-commentary-v12.de-en.de"
     },
    ]
SAVE_DATA_SRC = "bpe_vocab_src.pkl"
SAVE_DATA_TRG = "bpe_vocab_trg.pkl"
SAVE_DATA_TRAIN = "bpe_translate_train.pkl"
"""
Settings for BPE
"""
DATA_DIR = './data2/bpe/'
CODES = "codes.txt"
# symbols is the vocabulary size
SYMBOLS = 32000
MIN_FREQUENCY = 6
SEPARATOR = "@@"
PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
MAX_LEN = 100

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
    raw_files = { "src": [], "trg": [], }
    for d in sources:
        src_file, trg_file = extract(raw_dir, d['folder_name'], d["src"], d["trg"])
        raw_files["src"].append(src_file)
        raw_files["trg"].append(trg_file)
    return raw_files


def compile_files(raw_dir, raw_files, prefix):
    src_fpath = os.path.join(raw_dir, f"raw-{prefix}-src.txt")
    trg_fpath = os.path.join(raw_dir, f"raw-{prefix}-tgt.txt")

    if os.path.isfile(src_fpath) and os.path.isfile(trg_fpath):
        sys.stderr.write(f"Merged files found, skip the merging process.\n")
        return src_fpath, trg_fpath

    sys.stderr.write(f"Merge files into two files: {src_fpath} and {trg_fpath}.\n")

    with open(src_fpath, 'w') as src_outf, open(trg_fpath, 'w') as trg_outf:
        for src_inf, trg_inf in zip(raw_files['src'], raw_files['trg']):
            sys.stderr.write(f'  Input files: \n'\
                    f'    - SRC: {src_inf}, and\n' \
                    f'    - TRG: {trg_inf}.\n')
            with open(src_inf, newline='\n') as src_inf, open(trg_inf, newline='\n') as trg_inf:
                cntr = 0
                for i, line in enumerate(src_inf):
                    cntr += 1
                    src_outf.write(line.replace('\r', ' ').strip() + '\n')
                for j, line in enumerate(trg_inf):
                    cntr -= 1
                    trg_outf.write(line.replace('\r', ' ').strip() + '\n')
                assert cntr == 0, 'Number of lines in two files are inconsistent.'
    return src_fpath, trg_fpath


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


def main():
    raw_train = get_raw_files(RAW_DIR, _TRAIN_DATA_SOURCES)
    train_src, train_trg = compile_files(RAW_DIR, raw_train, PREFIX_NC + '-train')
    codes = os.path.join(DATA_DIR, CODES)
    learn_bpe(raw_train['src'] + raw_train['trg'], codes, SYMBOLS, MIN_FREQUENCY, True)
    with codecs.open(codes, encoding='utf-8') as codes:
        bpe = BPE(codes, separator=SEPARATOR)
    encode_files(bpe, train_src, train_trg, DATA_DIR, PREFIX_NC + '-train')
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

    enc_train_files_prefix = PREFIX_NC + '-train'
    train = TranslationDataset(
        fields=fields,
        path=os.path.join(DATA_DIR, enc_train_files_prefix),
        exts=('.src','.trg'),
        filter_pred=filter_examples_with_length)

    field_src.build_vocab(train.src, min_freq=2)
    field_trg.build_vocab(train.trg, min_freq=2)

    save_data_src = os.path.join(DATA_DIR, SAVE_DATA_SRC)
    save_data_trg = os.path.join(DATA_DIR, SAVE_DATA_TRG)
    save_data_train = os.path.join(DATA_DIR, SAVE_DATA_TRAIN)

    pickle.dump(field_src, open(save_data_src, 'wb'))
    pickle.dump(field_trg, open(save_data_trg, 'wb'))
    pickle.dump(train.examples, open(save_data_train, 'wb'))

if __name__ == '__main__':
    mkdir_if_needed(DATA_DIR)
    main()
