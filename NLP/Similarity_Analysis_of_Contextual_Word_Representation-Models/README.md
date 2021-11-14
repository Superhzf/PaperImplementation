# Similarity Analysis of Contextual Word Representation Models
With the instructions and code in this repo, you should be able to reproduce the content of the paper [Similarity Analysis of Contextual Word Representation Models](https://arxiv.org/pdf/2005.01172.pdf). I will focus on paper implementation instead of explanation. I run the instructions on MacOS Big Sur v11.6., it might be different on Windows systems.

# 1. Data Preparation
According to the authors, the data-set used in the paper is the development set of `Penn Treebank` data. You can obtain the raw data-set and preprocess using the `sanity_check.py` file by the command `python3 sanity_check.py` in your local Python environment.

The general idea of the process is to make sure sentences in the output are newline delimited, and tokens are space-delimited. Moreover, the file also ensures that for the same sentence, the `sequence_length` by different NLP models is the same for the sake of matrix multiplication.

The output of the file is `PennTreebank_develp_set.txt`.

# 2. Generate representations of the original ELM model
## 2.1. Environment setup
The authors of the paper obtain some representations based on the environment by the `Installation` part in this [repo](https://github.com/nelson-liu/contextual-repr-analysis). If you run into problems when doing the unit test by `py.test -v`, please make sure the package version is correct and it points to the Python in the conda environment instead of the system Python. If the package does not point to the Python in the conda environment, probably you have to edit the `$PATH` variable.

## 2.2. Generate representations

In the conda environment, you can obtain the representations of the original ELM model by running this command:

`allennlp elmo ./PennTreebank_develp_set.txt elmo_original.hdf5 --all --weight-file https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 --options-file https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json
`where `elmo_original.hdf5` is the output file that stores the representations. The file format follows this [description](https://github.com/nelson-liu/contextual-repr-analysis#step-1-precomputing-the-word-representations).

Basically, `all` means we want the output of all the three layers. For more details, you can refer to [this page](http://docs.allennlp.org/v0.9.0/api/allennlp.commands.elmo.html) for the command documentation, and [this page](https://allennlp.org/elmo) for more options of `--weight-file` and `--options-file`.

Starting from here, the file `elmo_original.hdf5` will be the standard reference for other models, and we will not use `PennTreebank_develp_set.txt`.

# 3. Generate representations of the ELM transformer model

## 3.1 Download the model
You can download the ELM transformer model (1.73GB) by opening [this page](https://allennlp.org/elmo), then `Contributed ELMo Models`, then `Transformer ELMo`.

## 3.2 Package modification

In order to finish this task, we have to modify the `allennlp` package in the conda environment (If you can find any easier to do it, please let me know), since `allennlp` package can only output the averaged representations of layers instead of all the middle layers separately.

1. We have to find out where the `allennlp` is installed by doing this:

```
import allennlp

allennlp.__file__
```

The output should similar to this
`/Users/yourname/opt/miniconda3/envs/contextual_repr_analysis/lib/python3.6/site-packages/allennlp/__init__.py`

2. Go to the `allennlp` folder by `cd /Users/yourname/opt/miniconda3/envs/contextual_repr_analysis/lib/python3.6/site-packages/allennlp/modules/token_embedders`

3. In the folder, edit the file by `vim language_model_token_embedder.py`, and in the file, set up the line numbers by `:set number` and we will start from line 172.

4. Add these lines below line 173
```
tensor_contextual_embed = torch.cat(contextual_embeddings,dim=0)
if self._remove_bos_eos:
  tensor_contextual_embed=tensor_contextual_embed[:,1:-1,:]
return tensor_contextual_embed
```
and comment out other lines that are related to `averaged_embeddings`.

The idea of doing this is instead of returning the averaged representations, we want the representations of all the middle layers. `1:-1` means I want to remove the special characters in the front and at the end.

5. Finally save the modification and exit.

## 3.3 Generate the representations

Now, you can generate the representations of the ELM transformer model in the conda environment by running `python elmo_transformer2HDF5.py`. The representations are stored in `elmo_transformer.hdf5`.

# 4. Generate representations of Bert, XLNet, GPT, and GPT2

We will do this task in the local Python environment mainly using the [huggingface](https://huggingface.co/) package instead of in the conda environment.

Doing this task is as simple as running `python3 transformer2HDF5.py`. The credits should go to [here](https://github.com/johnmwu/contextual-corr-analysis/blob/dev/get_transformer_representations.py)

# 5. Generate attentions of Bert, GPT2, and XLNet

Similarly, we will do this task in the local Python environment.

Doing this task is as simple as running `python3 attention2HDF5.py`. The credits should go to [here](https://github.com/johnmwu/contextual-corr-analysis/blob/dev/get_transformer_attentions.py)

# 6. Generate representations with random weights of GPT2 and XLNet

Again, we will do this task in the local Python environment.

Doing this task is as simple as running `python3 transformers2HDF5_rand.py`. The credits should go to [here](https://github.com/johnmwu/contextual-corr-analysis/blob/dev/get_transformer_representations.py)

# 7. Analysis

Now, we have collected all the data needed to reproduce the analysis in the paper.

Below I will refer you to the analysis file in the authors' Github repo. Basically, you calculate different statistical variables based on experiments [here](https://github.com/johnmwu/contextual-corr-analysis/tree/master/slurm), and you can analyze the variables and draw pictures based on the experiments [here](https://github.com/johnmwu/contextual-corr-analysis/tree/master/analysis).

In particular, if you want to reproduce Figure 1.(a) and 1.(b) in the paper, you have to focus on [experiment 14](https://github.com/johnmwu/contextual-corr-analysis/blob/master/slurm/mk_results14-helper.sh), and [analysis 14](https://github.com/johnmwu/contextual-corr-analysis/blob/master/analysis/analysis-14.ipynb).

Similarly, if you want to reproduce Figure 2 in the paper, you have to focus on experiment 13 or 15.

The idea to find out which experiment you should do is first finding out the pictures you want in the folder [experiment](https://github.com/johnmwu/contextual-corr-analysis/tree/master/analysis) among `*.png` files, and then find out which experiment generates this picture among `*.ipynb` files.
