import torch
from transformers import BertModel, BertConfig, DNATokenizer
import numpy as np
import pandas as pd
import csv

train_name = "m_b_train"
k_mer = 4
input_path = "./sample_data/m6a/"+str(k_mer)+"/"+train_name
output_hidden_path = "./last_hidden/"+str(k_mer)+"/"+train_name

dir_to_pretrained_model = "./fine_models/"+str(k_mer)+"/"+train_name

config = BertConfig.from_pretrained('https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-6/config.json')
config = BertConfig.from_pretrained('{}/config.json'.format(dir_to_pretrained_model))
tokenizer = DNATokenizer.from_pretrained('dna{}'.format(k_mer))
model = BertModel.from_pretrained(dir_to_pretrained_model, config=config)


train_path = input_path + "/" + "train_pri.tsv"
dev_path = input_path + "/" + "dev.tsv"

train = pd.read_csv(train_path, sep='\t', header=0)
train_data = train.loc[:, "sequence"].values
train_label = train.loc[:, "label"].values

train_state = []

dev = pd.read_csv(dev_path, sep="\t", header=0)
dev_data = dev.loc[:, "sequence"].values
dev_label = dev.loc[:, "label"].values

dev_state = []

for j in train_data:
    model_input = tokenizer.encode_plus(j, add_special_tokens=False, max_length=512)["input_ids"]
    model_input = torch.tensor(model_input, dtype=torch.long)
    model_input = model_input.unsqueeze(0)   # to generate a fake batch with batch size one
    output = model(model_input)
    train_state.append(list((np.array(output[1].detach().numpy()[0], dtype='float16')).reshape(1,-1))[0])

print(len(dev_data))
for j in dev_data:
    model_input = tokenizer.encode_plus(j, add_special_tokens=False, max_length=512)["input_ids"]
    model_input = torch.tensor(model_input, dtype=torch.long)
    model_input = model_input.unsqueeze(0)   # to generate a fake batch with batch size one
    output = model(model_input)
    # print((np.array(output[1].detach().numpy()[0], dtype='float16')).shape)
    dev_state.append(list((np.array(output[1].detach().numpy()[0], dtype='float16')).reshape(1, -1))[0])

train_state = pd.DataFrame(np.array(train_state))

train_state.to_csv('{}'.format(output_hidden_path) + '/train_hidden.csv', header=None)
dev_state = pd.DataFrame(np.array(dev_state))
print(dev_state.shape)
dev_state.to_csv('{}'.format(output_hidden_path) + '/dev_hidden.csv', header=None)
