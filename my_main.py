from __future__ import print_function, division
# import torch
from TextDataLoader import TextLoader
import os
import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle

import warnings
warnings.filterwarnings("ignore")
data_array = ['commoncrawl.de-en-de.txt', 'europarl-v7_de-en-de.txt', 'news-commentary-v10_de-en-de.txt'], \
    ['commoncrawl.de-en-en.txt', 'europarl-v7_de-en-en.txt', 'news-commentary-v10_de-en-en.txt']
root = 'lang_data/wmt15-de-en/'

sentences = pickle.load(open('paired_sentences.pkl', 'rb'))
sentence_pairs = sentences['data']

word2index_dict = pickle.load(open('word2index.pkl','rb'))
index2word_dict = pickle.load(open('index2word.pkl', 'rb'))



text_loader = TextLoader(sentence_pairs, word2index_dict, index2word_dict)

data_loader = DataLoader(text_loader, batch_size=4, shuffle=True, num_workers=4)

for i_batch, sample_batched in enumerate(data_loader):
	print(sample_batched)


