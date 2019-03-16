from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import os
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda")

SOS_token = 0
EOS_token = 1



class language(object):
    """docstring for language"""
    def __init__(self, name):
        super(language, self).__init__()
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)


    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

        

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_lang_data(lang1, lang2):
    root = 'lang_data/wmt15-de-en/e'
    lang1_data = []
    for data_file in lang1:
        data = open(os.path.join(root,data_file), encoding='utf-8').read().strip().split('\n')
        lang1_data.extend(data)
    
    lang2_data = []
    for data_file in lang2:
        data = open(os.path.join(root,data_file), encoding='utf-8').read().strip().split('\n')
        lang2_data.extend(data)

    pairs = [[normalizeString(l1) ,normalizeString(l2)] for l1, l2 in zip(lang1_data, lang2_data)]

    language_1 = language('German')
    language_2 = language('English')

    return language_1, language_2, pairs

def prepare_dataset():
    lang_1, lang_2, pairs = read_lang_data(['commoncrawl.de-en-de.txt', 'europarl-v7_de-en-de.txt', 'news-commentary-v10_de-en-de.txt'], \
    ['commoncrawl.de-en-en.txt', 'europarl-v7_de-en-en.txt', 'news-commentary-v10_de-en-en.txt'])

    for pair in pairs:
        lang_1.addSentence(pair[0])
        lang_2.addSentence(pair[1])
    print(lang_1.name, lang_1.n_words)
    print(lang_2.name, lang_2.n_words)
    
    return lang_1, lang_2, pairs






read_lang_data(['commoncrawl.de-en-de.txt', 'europarl-v7_de-en-de.txt', 'news-commentary-v10_de-en-de.txt'], \
    ['commoncrawl.de-en-en.txt', 'europarl-v7_de-en-en.txt', 'news-commentary-v10_de-en-en.txt'])