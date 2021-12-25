# ref: https://www.statmt.org/wmt17/translation-task.html#download
import os
import re
import string
data_source = "./data/"
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
def read_file(folder_name,file_name,file_type):
    lines=[]
    with open("{}/{}".format(folder_name,file_name)) as file:
        print("reading {}".format(file_name))
        for line in file:
            # Not clear what preprocessing steps are used, need confirmation from the authors.
#             line = remove_special_characters(line)
#             line = remove_punctuation(line)
#             line = line.strip()
#             line = remove_redundant_space(line)
            if line != "":
                lines.append(line.lower())
            else:
                "{} in {} is skipped!".format(line, file_name)
    return lines

def flat_data(nested_dataset):
    flat_dataset=[]
    for sublist in nested_dataset:
        for item in sublist:
            flat_dataset.append(item)
    return flat_dataset

if __name__ == '__main__':
    nested_en_dataset = []
    nested_de_dataset = []
    for this_folder in [x[0] for x in os.walk(data_source)]:
        lines_en = []
        lines_de = []
        for file in os.listdir(this_folder):
            if file.endswith("de-en.en"):
                lines_en = read_file(this_folder,file,"de-en.en")
                nested_en_dataset.append(lines_en)
                print("Finished reading {} sentences from {}".format(len(lines_en),file))
            elif file.endswith("de-en.de"):
                lines_de = read_file(this_folder,file,"de-en.de")
                nested_de_dataset.append(lines_de)
                print("Finished reading {} sentences from {}".format(len(lines_de),file))


    flat_en_dataset = flat_data(nested_en_dataset)
    flat_de_dataset = flat_data(nested_de_dataset)
    with open('{}preprocessed_en.txt'.format(data_source), 'w') as en_file:
        for sentence in flat_en_dataset:
            en_file.write(sentence)
    en_file.close()
    print(f"Finished writing {len(flat_en_dataset)} English sentences into a file")
    with open('{}preprocessed_de.txt'.format(data_source), 'w') as de_file:
        for sentence in flat_de_dataset:
            de_file.write(sentence)
    en_file.close()
    print(f"Finished writing {len(flat_de_dataset)} German sentences into a file")
