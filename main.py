import sys
import numpy as np
import json
import torchtext
from torchtext import data, datasets

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from model import attbilstm
from prepro import load_data_and_labels

with open('config.json', 'r') as fo:
    config = json.load(fo)
print(config)

import random
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

TEXT = data.Field(sequential=True, tokenize=lambda x:x.split(), lower=True)
LABEL = data.LabelField(use_vocab=True)

dataset  = data.TabularDataset(path='all.csv', format='csv', fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
train_dt, valid_dt, test_dt = dataset.split(split_ratio=[0.7, 0.1, 0.2], random_state=random.getstate())
TEXT.build_vocab(train_dt)
LABEL.build_vocab(train_dt)

bsize = config["batch_size"]
gpu = config["gpu"]
device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
train_it, valid_it, test_it = data.BucketIterator.splits((train_dt, valid_dt, test_dt),batch_sizes=(bsize,bsize,bsize), device=device, sort_key=lambda x: len(x.text), repeat=False)

'''
for b in train_it:
    print (b.text, b.label)
    sys.exit()
'''
def accuracy(pred, label):
    prob, idx = torch.max(pred,1)
    precision = (idx==label).float().mean()
    if gpu:
        accuracy.append(precision.data.item())
    else:
        accuracy.append(precision.data.numpy()[0])
    return np.mean(accuracy)

def cat_accuracy(pred, label):
    max_preds = pred.argmax(dim=1, keepdim=True)
    correct = max_preds.squeeze(1).eq(label)
    #print (correct.sum)
    #print (torch.LongTensor([label.shape[0]]))
    return correct.sum()/torch.LongTensor([label.shape[0]])

def train(model, it, lossf, optimizer):
    model.train()
    ep_loss = 0.0
    ep_acc = 0
    for b in it:
        optimizer.zero_grad()
        seq, label = b.text, b.label
        pred = model(seq)
        #print (pred)
        #print (pred, label)
        loss = lossf(pred, label)
        acc = cat_accuracy(pred, label)
        loss.backward()
        optimizer.step()
        ep_loss += loss.item()
        ep_acc += acc.item()
    return ep_loss/ len(it), ep_acc/ len(it)


vocab_size = len(TEXT.vocab)
model = attbilstm(vocab_size, config)
#lossf = nn.BCEWithLogitsLoss()
lossf = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

if gpu:
    model.to(device)

for ep in range(config['epochs']):
    print ('training...')
    tr_loss, tr_acc = train(model, train_it, lossf, optimizer)
    #print ('TRAIN: loss %.2f' % (tr_loss))
    print ('TRAIN: loss %.2f acc %.1f' % (tr_loss, tr_acc*100))

