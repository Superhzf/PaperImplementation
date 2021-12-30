import os
from io import open
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, development_mode=False):
        self.dictionary = Dictionary()
        # TODO: currently, it only works for LM problem.
        if not development_mode:
            self.train = self.tokenize(os.path.join(path, 'preprocessed_en_trn.txt'))
            self.val = self.tokenize(os.path.join(path, 'preprocessed_en_val.txt'))
        else:
            self.train = self.tokenize(os.path.join(path, 'develop_train.txt'))
            self.val = self.tokenize(os.path.join(path, 'develop_valid.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

class CorpusMT:
    def __init__(self, path, development_mode=False):
        self.train = self.read_data(path, 'preprocessed_en_trn.txt', 'preprocessed_de_trn.txt',development_mode)
        self.val = self.read_data(path, 'preprocessed_en_val.txt', 'preprocessed_de_val.txt',development_mode)

    def read_data(self,path, file_name_src,file_name_tgt,development_mode=False):
        data_set = []
        src_file = os.path.join(path, file_name_src)
        tgt_file = os.path.join(path, file_name_tgt)
        with open(src_file,'r') as f:
            src = f.readlines()
        f.close()
        with open(tgt_file,'r') as f:
            tgt = f.readlines()
        f.close()
        #TODO: the length of two files should be the same
        min_len = min(len(src),len(tgt))
        if development_mode:
            min_len = int(0.01*min_len)
        for i in range(min_len):
            this_group = (src[i],tgt[i])
            data_set.append(this_group)
        return data_set

class DatasetMLM(Dataset):
    def __init__(self, src, tokenizer):
        self.src = src
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = self.src[idx]
        return src

class CorpusMLM:
    def __init__(self,path, development_mode=False):
        self.train = self.read_data(os.path.join(path, 'preprocessed_en_trn.txt'),development_mode,True)
        self.val = self.read_data(os.path.join(path, 'preprocessed_en_val.txt'),development_mode,False)

    def read_data(self, path,development_mode=False,training=True):
        if training:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_basic_tokenization = True)
        with open(path,'r') as f:
            src = f.readlines()
        f.close()
        if development_mode:
            src_len = int(len(src)*0.01)
            src = src[:src_len]
        this_dataset = DatasetMLM(src,self.tokenizer)
        return this_dataset
