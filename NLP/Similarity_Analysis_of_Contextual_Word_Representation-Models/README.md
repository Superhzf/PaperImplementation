# Similarity Analysis of Contextual Word Representation Models
With the instructions and code in this repo, you should be able to reproduce the content of the paper [Similarity Analysis of Contextual Word Representation Models](https://arxiv.org/pdf/2005.01172.pdf). I will focus on paper implementation instead of explanation. I run the instructions on MacOS Big Sur v11.6., it might be different on Windows systems.

# 1. Data Preparation
According to the authors, the data-set used in the paper is the development set of `Penn Treebank` data. You can obtain the raw data-set and preprocess using the `sanity_check.py` file by the command `python3 sanity_check.py`.

The general idea of the process is to make sure sentences in the output are newline delimited, and tokens are space-delimited. Moreover, the file also ensures that for the same sentence, the `sequence_length` by different NLP models is the same for the sake of matrix multiplication.

The output of the file is `PennTreebank_develp_set.txt`, which is ready for generating representation files by different NLP models.

# 2. Generate representations of the original ELM model
## 2.1. Environment setup
The authors of the paper set up the environment by following the `Installation` part in this [repo](https://github.com/nelson-liu/contextual-repr-analysis). If you run into problems when doing the unit test by `py.test -v`, please make sure the package version is correct and it points to the Python in the environment instead of the system Python. If the package does not point to the Python in the environment, probably you have to edit the `$PATH` variable.

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

In order to finish this task, we have to modify the `allennlp` package in the conda environment (If you can find any easier to do it, please let me know), since `allennlp` package can only output the representations of the final layer instead of all the middle layers.

1. we have to find out where the `allennlp` is installed by doing this:

```
import allennlp

allennlp.__file__
```

The output should similar to this
`/Users/yourname/opt/miniconda3/envs/contextual_repr_analysis/lib/python3.6/site-packages/allennlp/__init__.py`

2. Go to the `allennlp` folder by `cd /Users/yourname/opt/miniconda3/envs/contextual_repr_analysis/lib/python3.6/site-packages/allennlp/modules/token_embedders`

3. In the folder, edit the file by `vim language_model_token_embedder.py`, and in the file, set up the line numbers by `set number` and we will start from line 172.

4. 
`
