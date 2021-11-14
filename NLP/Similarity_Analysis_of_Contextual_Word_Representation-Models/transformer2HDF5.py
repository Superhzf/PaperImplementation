# https://mccormickml.com/2019/09/19/XLNet-fine-tuning/
# https://medium.com/swlh/using-xlnet-for-sentiment-classification-cfa948e65e85
import json
import h5py
import torch
from transformers import BertTokenizer, BertModel, XLNetTokenizer, XLNetModel,GPT2Tokenizer, GPT2Model, OpenAIGPTTokenizer, OpenAIGPTModel
import numpy as np

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
        print('Randomizing weights')
        model.init_weights()

    return model, tokenizer, sep

def get_sentence_repr(sentence, model, tokenizer, sep, model_name,include_embeddings=False):
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
        all_hidden_states = model(input_ids)[-1]
        if include_embeddings:
            all_hidden_states = [hidden_states[0].cpu().numpy() for hidden_states in all_hidden_states]
        else:
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

model_names = ["bert-base-cased","bert-large-cased","gpt2","gpt2-medium","xlnet-large-cased","xlnet-base-cased", "openai-gpt"]
total_num = len(sentence_to_idx.items())
for model_name in model_names:
    hdf5_file = '{}.hdf5'.format(model_name)
    print("Start working on {} model".format(model_name))
    model, tokenizer, sep = get_model_and_tokenizer(model_name, random_weights=False)
    for sentence,idx in sentence_to_idx.items():
        if int(idx)%100 == 0 and int(idx)>0:
            print("{} model {:.2f} percent has been finished!".format(model_name, 100*int(idx)/total_num))
        hidden_states = get_sentence_repr(sentence, model, tokenizer, sep, model_name,include_embeddings=False)
        result_sentence2idx[sentence] = idx
        result_idx2embed[idx] = hidden_states
        # if int(idx) == 3:
        #     break

    make_hdf5_file(hdf5_file,result_sentence2idx,result_idx2embed)
    print("Successfully write {} model repre to HDF5 file".format(model_name))
