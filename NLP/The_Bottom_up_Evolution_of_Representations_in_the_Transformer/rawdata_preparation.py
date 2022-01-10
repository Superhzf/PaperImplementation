from datasets import load_dataset
import os
import sys
from data_preprocessing2 import mkdir_if_needed

DATA_FOLDER = './data/raw'
VALID_EN = 'valid_en.txt'
VALID_DE = 'valid_de.txt'
TRAIN_EN_FULL = 'train_en_full.txt'
TRAIN_DE_FULL = 'train_de_full.txt'
TRAIN_EN_DEV = 'train_en_dev.txt'
TRAIN_DE_DEV = 'train_de_dev.txt'


def write2file(data_folder, dataset, src_file_name, trg_file_name, generate_data_dev_mode=False):
    """
    The function to write src and trg sentences into separate files.
    ----------------------------------------------------
    Parameter:
    data_folder: str
        Where you want to store the saved dataset
    dataset: datasets.arrow_dataset.Dataset
        The dataset from huggingface that includes both the src and the trg data
    src_file_name: str
        The file name of the source language.
    trg_file_name: str
        The file name of the target language.
    generate_data_dev_mode: bool
        If true, only the first 5000 sentences will be written into text files.
    """
    with open(os.path.join(data_folder, src_file_name), 'w') as src_outf,\
    open(os.path.join(data_folder, trg_file_name), 'w') as trg_outf:
        for index, this_sample in enumerate(dataset.data[0]):
            src_outf.write(this_sample['en'].as_py())
            src_outf.write('\n')
            trg_outf.write(this_sample['de'].as_py())
            trg_outf.write('\n')
            if generate_data_dev_mode and index>=5000:
                break;
    src_outf.close()
    trg_outf.close()
    sys.stderr.write(f"{index+1} sentences have been written to {os.path.join(data_folder, src_file_name)}\
     and {os.path.join(data_folder, trg_file_name)}")
    sys.stderr.write('\n')


def main():
    # https://huggingface.co/docs/datasets/package_reference/loading_methods.html#datasets.load_dataset
    # It may take about 20 mins to download the dataset for the first time.
    dataset = load_dataset(path='wmt17',name="de-en",split=['train','validation'])
    write2file(DATA_FOLDER, dataset[1], VALID_EN, VALID_DE)
    write2file(DATA_FOLDER, dataset[0], TRAIN_EN_FULL, TRAIN_DE_FULL)
    write2file(DATA_FOLDER, dataset[0], TRAIN_EN_DEV, TRAIN_DE_DEV, True)

if __name__ == '__main__':
    mkdir_if_needed(DATA_FOLDER)
    main()
