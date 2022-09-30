# Implement Resnet using Pytorch from scratch

This folder aims to implement the code and reproduce the results of the vanilla
[resnet paper](https://arxiv.org/abs/1512.03385) on the CIFAR 10 dataset using Pytorch 1.12.1 with CUDA 11.3.

Generally, I follow the setups mentioned by the paper to my best knowledge (including # of layers, filter size, epochs, augmentation, and so on so forth).

As a result, this implementation can see `90.51%` accuracy on the test set based on the setups below. If normalizing the dataset using the accurate mean and std, probably it will further improve the accuracy.

`log_2022` is the log file using `seed=2022` and `log_2023` is the log file using `seed=2023`. `log_2022_v2` is the log file running on the bottleneck block with `seed=2022`. You may have to write extra code to generate the log file yourself. I run the code on a machine using `sbatch` and the log file is generated automatically.


# Usage

In the current folder, run:

`python3 main.py -seed 2022 -resnet_version 1 -resnet_size 3 -num_classes 10 -num_epochs 164 -batch 128 -lr 0.1 -drop 0`

where

`resnet_version = 1` indicates the vanilla residual block and `resnet_version = 2` indicates the bottleneck block.


`resnet_size` decides the total layers. The total layers equal `6*resnet_size + 2` for version 1 and `9*resnet_size + 2` for version 2.

# Important parameters and preprocessing:

Per the paper, it runs ~64k iterations with the batch size 128. Therefore, the total `epochs` equal 64000//(50000//128) = 164.

`lr` is divided by 10 at 32000 and 48000 iterations. Translating to epochs, the `lr` rate is divided by 10 at 32000//(50000//128) = 82th epoch and 48000//(50000//128) = 123th epoch.

`data augmentation` is executed by padding 4 pixels and 32x32 crop is randomly sampled from the padded image or its horizontal flip on the training set. Only normalization is applied to the test set.

# Insights

I try my best to follow the setups by the paper. But using the accurate mean and std for normalization will massively slow down the training process. This is why I use `mean = [0.5, 0.5, 0.5]` and `std = [0.5,0.5,0.5]` as the mean and std vectors.

With or without data augmentation affect the accuracy on the test set by `~6%`.

# ETA

Running on a single RTX 6000 GPU, training the model above takes less than 30 minutes. Each epoch takes ~10 seconds.
