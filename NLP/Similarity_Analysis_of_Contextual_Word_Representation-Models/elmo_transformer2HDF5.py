from allennlp.modules.token_embedders.bidirectional_language_model_token_embedder import BidirectionalLanguageModelTokenEmbedder
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers.token import Token
import torch
import json
import h5py

# this should run in the miniconda
# Allennlp package should be modified, find the location by allennlp.__file__
# can also refer to this: https://github.com/allenai/allennlp/commit/088f0bb685231a17320bf916c091dfa8e60e3ce5

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


lm_model_file = "./transformer-elmo-2019.01.10.tar.gz"
hdf5_file = 'elmo_transformer.hdf5'
source_file = "./HDF5files/elmo_original.hdf5"
activations_h5 = h5py.File(source_file, 'r')
sentence_to_idx = json.loads(activations_h5['sentence_to_index'][0])
activations_h5.close()
key = 0
result_idx2embed={}
result_sentence2idx={}

lm_embedder = BidirectionalLanguageModelTokenEmbedder(
  archive_file=lm_model_file,
  dropout=0.2,
  bos_eos_tokens=["<S>", "</S>"],
  remove_bos_eos=True,
  requires_grad=False
)

indexer = ELMoTokenCharactersIndexer()
vocab = lm_embedder._lm.vocab
total_num = len(sentence_to_idx.items())
for sentence,idx in sentence_to_idx.items():
# test_sentence = "It is no longer snowing in Munich ."
    if int(idx)%100 == 0 and int(idx)>0:
        print("{:.2f} percent has been finished!".format(100*int(idx)/total_num))
    tokens = [Token(word) for word in sentence.split()]

    character_indices = indexer.tokens_to_indices(tokens, vocab, "elmo")["elmo"]
    indices_tensor = torch.LongTensor([character_indices])
    # Embed and extract the single element from the batch.
    embeddings_fulls = lm_embedder(indices_tensor)

    result_idx2embed[idx]=embeddings_fulls
    result_sentence2idx[sentence]=str(key)
    result_sentence2idx[sentence]=idx
    key+=1
    # if key == 1:
    #     break

print("Finished calculating embeddings!")
# result_sentence2idx = json.dumps(result_sentence2idx)
print("Start the dumping HDF5 file!")
make_hdf5_file(hdf5_file,result_sentence2idx,result_idx2embed)
print("Successful!")
