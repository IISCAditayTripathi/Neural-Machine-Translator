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
from tqdm import tqdm 

import warnings
warnings.filterwarnings("ignore")
data_array = ['commoncrawl.de-en-de.txt', 'europarl-v7_de-en-de.txt', 'news-commentary-v10_de-en-de.txt'], \
    ['commoncrawl.de-en-en.txt', 'europarl-v7_de-en-en.txt', 'news-commentary-v10_de-en-en.txt']
root = 'lang_data/wmt15-de-en/'

sentences = pickle.load(open('paired_sentences.pkl', 'rb'))
sentence_pairs = sentences['data']

word2index_dict = pickle.load(open('word2index_v3.pkl','rb'))
index2word_dict = pickle.load(open('index2word_v3.pkl', 'rb'))



text_loader = TextLoader(sentence_pairs, word2index_dict, index2word_dict)

data_loader = DataLoader(text_loader, batch_size=20, shuffle=True, num_workers=25)

loader = tqdm(data_loader, total=len(data_loader), unit="batches")

for i_batch, sample_batched in enumerate(loader):
	loader.set_postfix(Number= i_batch)
