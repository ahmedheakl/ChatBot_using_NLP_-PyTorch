from nltk_utils import tokenize, stem, bag_of_words
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# get data from json
with open('intents.json', 'r') as f:
    intents = json.load(f)

#  apply the utils
# print(intents)
tags = []
all_words = []
pattern_tag = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        sentence = tokenize(pattern)
        all_words.extend(sentence)
        pattern_tag.append((sentence, tag))

# Stem and Remove punctuation
punc_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in punc_words]


# sort and remove redundent words
all_words = sorted(set(all_words))
tags = sorted(tags)
# print(all_words)

# Make X_train, y_train
X_train = []
y_train = []
for (pattern, tag) in pattern_tag:
    bag = bag_of_words(pattern, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# make the dataset class


class myDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.n_samples = len(y_train)

    def __getitem__(self, index):
        return self.X_train[index], self.y_train[index]

    def __len__(self):
        return self.n_samples


#  Setting the data and saving it
dataset = myDataset(X_train, y_train)
torch.save(dataset, './dataset.pt')
dd = torch.load('./dataset.pt')
