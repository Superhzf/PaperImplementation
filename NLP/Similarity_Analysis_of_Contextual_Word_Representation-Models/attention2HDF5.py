# https://mccormickml.com/2019/09/19/XLNet-fine-tuning/
# https://medium.com/swlh/using-xlnet-for-sentiment-classification-cfa948e65e85
import json
import h5py
import torch
from transformers import BertTokenizer, BertModel, XLNetTokenizer, XLNetModel,GPT2Tokenizer, GPT2Model
import numpy as np

def remove_special_characters(text):
    # define the pattern to keep
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]'
    return re.sub(pat, ' ', text)
def remove_punctuation(text):
    text = ''.join([c for c in text if c not in string.punctuation])
    return text
def remove_numbers(text):
    # define the pattern to keep
    pattern = r'[^a-zA-z.,!?/:;\"\'\s]'
    return re.sub(pattern, '', text)
def remove_redundant_space(text):
    return " ".join(text.split())

def get_model_and_tokenizer(model_name, random_weights=False):

    if model_name.startswith('xlnet'):
        model = XLNetModel.from_pretrained(model_name, output_attentions=True)
        tokenizer = XLNetTokenizer.from_pretrained(model_name)
        sep = u'▁'
    elif model_name.startswith('gpt2'):
        model = GPT2Model.from_pretrained(model_name, output_attentions=True)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        sep = 'Ġ'
    elif model_name.startswith('bert'):
        model = BertModel.from_pretrained(model_name, output_attentions=True)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        sep = '##'
    else:
        print('Unrecognized model name:', model_name)
        sys.exit()

    if random_weights:
        print('Randomizing weights')
        model.init_weights()

    return model, tokenizer, sep

def get_sentence_repr(sentence, model, tokenizer, sep, model_name):
    """
    Get representations for one sentence
    """

    with torch.no_grad():
        ids = tokenizer.encode(sentence)
        if model_name.startswith('bert'):
            ids = ids[1:-1]
        elif model_name.startswith('xlnet'):
            ids = ids[:-2]
        input_ids = torch.tensor([ids])
        all_attentions = model(input_ids)[-1]
        # squeeze batch dimension --> numpy array of shape (num_layers, num_heads, sequence_length, sequence_length)
        all_attentions = np.array([attention[0].cpu().numpy() for attention in all_attentions])

    #For each word, take the representation of its last sub-word
    segmented_tokens = tokenizer.convert_ids_to_tokens(ids)
    assert len(segmented_tokens) == all_attentions.shape[2], 'incompatible tokens and states'
    # convert subword attention to word attention
    word_to_subword = get_word_to_subword(segmented_tokens, sep, model_name)
    all_attentions = [[get_word_word_attention(attention_h, word_to_subword) for attention_h in attention_l] for attention_l in all_attentions]
    all_attentions = np.array(all_attentions)

    return all_attentions

def get_word_to_subword(segmented_tokens, sep, model_name):
    """
    return a list of lists, where each element in the top list is a word and each nested list is indices of its subwords
    """

    word_to_subword = []

    # example: ['Jim', 'ĠHend', 'riks', 'Ġis', 'Ġa', 'Ġpupp', 'ete', 'er']
    # output: [[0], [1, 2], [3], [4], [5,6,7]]
    if model_name.startswith('gpt2') or model_name.startswith('xlnet'):
        cur_word = []
        for i in range(len(segmented_tokens)):
            if segmented_tokens[i].startswith(sep):
                # don't append an empty list (may happen when i = 0)
                if len(cur_word) > 0:
                    word_to_subword.append(cur_word)
                cur_word = [i]
            else:
                cur_word.append(i)
        word_to_subword.append(cur_word)

    # example: ['Jim', 'He', '##nd', '##rik', '##s', 'is', 'a', 'puppet', '##eer']
    # output: [[0], [1,2,3,4], [5], [6], [7], [8,9]]
    elif model_name.startswith('bert'):
        cur_word = []
        for i in range(len(segmented_tokens)):
            if not segmented_tokens[i].startswith(sep):
                if len(cur_word) > 0:
                    word_to_subword.append(cur_word)
                cur_word = [i]
            else:
                cur_word.append(i)
        word_to_subword.append(cur_word)

    else:
        raise ValueError('Unrecognized model name:', model_name)

    return word_to_subword

# modified from Clark et al. 2019, What Does BERT Look At? An Analysis of BERT's Attention
def get_word_word_attention(token_token_attention, words_to_tokens, mode="mean"):
    """Convert token-token attention to word-word attention (when tokens are
    derived from words using something like byte-pair encodings)."""

    #print(token_token_attention)
    #print(words_to_tokens)

    word_word_attention = np.array(token_token_attention)
    not_word_starts = []
    for word in words_to_tokens:
        not_word_starts += word[1:]

    # sum up the attentions for all tokens in a word that has been split
    for word in words_to_tokens:
        #print(word)
        word_word_attention[:, word[0]] = word_word_attention[:, word].sum(axis=-1)
    word_word_attention = np.delete(word_word_attention, not_word_starts, -1)

    # several options for combining attention maps for words that have been split
    # we use "mean" in the paper
    for word in words_to_tokens:
        if mode == "first":
            pass
        elif mode == "mean":
            word_word_attention[word[0]] = np.mean(word_word_attention[word], axis=0)
        elif mode == "max":
            word_word_attention[word[0]] = np.max(word_word_attention[word], axis=0)
            word_word_attention[word[0]] /= word_word_attention[word[0]].sum()
        else:
            raise ValueError("Unknown aggregation mode", mode)
    word_word_attention = np.delete(word_word_attention, not_word_starts, 0)

    return word_word_attention


def make_hdf5_file(output_file_path, sentence_to_index, vectors):
    with h5py.File(output_file_path, 'w') as fout:
        for key, embeddings in vectors.items():
            fout.create_dataset(
                str(key),
                embeddings.shape, dtype='float32',
                data=embeddings)
        sentence_index_dataset = fout.create_dataset(
            "sentence_to_index",
            (1,),
            dtype=h5py.special_dtype(vlen=str))
        sentence_index_dataset[0] = json.dumps(sentence_to_index)

result_idx2embed={}
result_sentence2idx = {}
source_file = "./HDF5files/elmo_original.hdf5"
activations_h5 = h5py.File(source_file, 'r')
sentence_to_idx = json.loads(activations_h5['sentence_to_index'][0])
activations_h5.close()

model_names = ["bert-base-cased","bert-large-cased","gpt2","gpt2-medium","xlnet-large-cased","xlnet-base-cased"]
total_num = len(sentence_to_idx.items())
for model_name in model_names:
    hdf5_file = '{}_attn.hdf5'.format(model_name)
    print("Start working on {} model".format(model_name))
    model, tokenizer, sep = get_model_and_tokenizer(model_name, random_weights=False)
    for sentence,idx in sentence_to_idx.items():
        if int(idx)%100 == 0 and int(idx)>0:
            print("{} model {:.2f} percent has been finished!".format(model_name, 100*int(idx)/total_num))
        attentions = get_sentence_repr(sentence, model, tokenizer, sep, model_name)
        result_sentence2idx[sentence] = idx
        result_idx2embed[idx] = attentions

    make_hdf5_file(hdf5_file,result_sentence2idx,result_idx2embed)
    print("Successfully write {} model repre to HDF5 file".format(model_name))
