import torch
from transformers import BertTokenizer, BertModel, XLNetTokenizer, XLNetModel,GPT2Tokenizer, GPT2Model, OpenAIGPTTokenizer, OpenAIGPTModel
import numpy as np
from torchtext.datasets import PennTreebank
import re
import string
# from contractions import CONTRACTION_MAP
# https://towardsdatascience.com/nlp-building-text-cleanup-and-preprocessing-pipeline-eba4095245a0
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
        model = XLNetModel.from_pretrained(model_name, output_hidden_states=True)
        tokenizer = XLNetTokenizer.from_pretrained(model_name)
        sep = u'▁'
    elif model_name.startswith('gpt2'):
        model = GPT2Model.from_pretrained(model_name, output_hidden_states=True)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        sep = 'Ġ'
    elif model_name.startswith('openai-gpt'):
        model = OpenAIGPTModel.from_pretrained(model_name, output_hidden_states=True)
        tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name)
        sep = '</w>'
    elif model_name.startswith('bert'):
        model = BertModel.from_pretrained(model_name, output_hidden_states=True)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        sep = '##'
    elif model_name.startswith('roberta'):
        model = RobertaModel.from_pretrained(model_name, output_hidden_states=True)
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        sep = 'Ġ'
    else:
        print('Unrecognized model name:', model_name)
        sys.exit()

    if random_weights:
        model.apply(model.init_weights)

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
        try:
            all_hidden_states = model(input_ids)[-1]
        except:
            print("sentence",sentence)
            print("input_ids",input_ids)
            exit(0)
        all_hidden_states = [hidden_states[0].cpu().numpy() for hidden_states in all_hidden_states[:-1]]
        all_hidden_states = np.array(all_hidden_states)

    #For each word, take the representation of its last sub-word
    segmented_tokens = tokenizer.convert_ids_to_tokens(ids)
    assert len(segmented_tokens) == all_hidden_states.shape[1], 'incompatible tokens and states'
    mask = np.full(len(segmented_tokens), False)

    if model_name.startswith('gpt2') or model_name.startswith('xlnet') or model_name.startswith('roberta'):
        # if next token is a new word, take current token's representation
        #print(segmented_tokens)
        for i in range(len(segmented_tokens)-1):
            if segmented_tokens[i+1].startswith(sep):
                #print(i)
                mask[i] = True
        # always take the last token representation for the last word
        mask[-1] = True
    # example: ['jim</w>', 'henson</w>', 'was</w>', 'a</w>', 'pup', 'pe', 'teer</w>']
    elif model_name.startswith('openai-gpt'):
        # if current token is a new word, take it
        for i in range(len(segmented_tokens)):
            if segmented_tokens[i].endswith(sep):
                mask[i] = True
        mask[-1] = True
    elif model_name.startswith('bert'):
        # if next token is not a continuation, take current token's representation
        for i in range(len(segmented_tokens)-1):
            if not segmented_tokens[i+1].startswith(sep):
                mask[i] = True
        mask[-1] = True
    else:
        print('Unrecognized model name:', model_name)
        sys.exit()

    all_hidden_states = all_hidden_states[:, mask]

    return all_hidden_states

_,development_set,_ = PennTreebank(root='.data', split=('train', 'valid', 'test'))
model_name1='bert-base-cased'
model_name2='gpt2'
model_name3='xlnet-base-cased'
model_name4='openai-gpt'
model1, tokenizer1, sep1 = get_model_and_tokenizer(model_name1)
model2, tokenizer2, sep2 = get_model_and_tokenizer(model_name2)
model3, tokenizer3, sep3 = get_model_and_tokenizer(model_name3)
model4, tokenizer4, sep4 = get_model_and_tokenizer(model_name4)
total_num = len(development_set)
lines_seen = set()
with open('PennTreebank_develp_set.txt', 'w') as the_file:
    for idx, sentence in enumerate(development_set):
        if idx%100 == 0 and idx>0:
            print("{:.2f}% has been finished!".format(100*idx/total_num))
        sentence = sentence.replace("<unk>","")
        sentence = remove_special_characters(sentence)
        sentence = remove_punctuation(sentence)
        sentence = sentence.strip()
        sentence = remove_redundant_space(sentence)
        if len(sentence)>=1:
            hidden_states1 = get_sentence_repr(sentence, model1, tokenizer1, sep1, model_name1)
            hidden_states2 = get_sentence_repr(sentence, model2, tokenizer2, sep2, model_name2)
            hidden_states3 = get_sentence_repr(sentence, model3, tokenizer3, sep3, model_name3)
            hidden_states4 = get_sentence_repr(sentence, model4, tokenizer4, sep4, model_name4)
            if (hidden_states1.shape[1] != hidden_states2.shape[1] or
                hidden_states2.shape[1] != hidden_states3.shape[1] or
                hidden_states3.shape[1] != hidden_states4.shape[1]):
                print("sentence:",sentence)
                print("hidden_states1.shape",hidden_states1.shape,
                        "hidden_states2.shape",hidden_states2.shape,
                        "hidden_states3.shape",hidden_states3.shape,"hidden_states4.shape",hidden_states4.shape)
                break
            if sentence not in lines_seen: # not a duplicate
                the_file.write(sentence+'\n')
                lines_seen.add(sentence)

the_file.close()
print("Total lines written:",len(lines_seen))
