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
from torchvision import transforms, utils
import string
from tqdm import tqdm
try:
    maketrans = ''.maketrans
except AttributeError:
    # fallback for Python 2
    from string import maketrans
# from string import maketrans
import pickle
from collections import defaultdict

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
    # table = string.maketrans(",")
    s = s.translate(maketrans("","", string.punctuation))
    return s


def read_lang_data(lang1, lang2):
    root = 'lang_data/wmt15-de-en/'
    lang1_data = []
    for data_file in lang1:
        data = open(os.path.join(root,data_file), encoding='utf-8').read().strip().split('\n')
        lang1_data.extend(data)
    
    lang2_data = []
    for data_file in lang2:
        data = open(os.path.join(root,data_file), encoding='utf-8').read().strip().split('\n')
        lang2_data.extend(data)

    # pairs = [[normalizeString(l1) ,normalizeString(l2)] for l1, l2 in zip(lang1_data, lang2_data)]
    pairs = []
    for i in tqdm(zip(lang1_data, lang2_data)):
        pairs.append([normalizeString(i[0]), normalizeString(i[1])])

    return  pairs

# def prepare_dataset():
#      pairs = read_lang_data(['commoncrawl.de-en-de.txt', 'europarl-v7_de-en-de.txt', 'news-commentary-v10_de-en-de.txt'], \
#     ['commoncrawl.de-en-en.txt', 'europarl-v7_de-en-en.txt', 'news-commentary-v10_de-en-en.txt'])

#     for pair in pairs:
#         lang_1.addSentence(pair[0])
#         lang_2.addSentence(pair[1])
#     print(lang_1.name, lang_1.n_words)
#     print(lang_2.name, lang_2.n_words)
    
#     return lang_1, lang_2, pairs

def get_lang_dict(sentence_pairs):
    word2index_lang1 = defaultdict(int)
    word2index_lang2 = defaultdict(int)
    index2word_lang1 = defaultdict(int)
    index2word_lang2 = defaultdict(int)
    index2word_lang1[0] = 'SOS'
    index2word_lang1[1] = 'EOS'

    index2word_lang2[0] = 'SOS'
    index2word_lang2[1] = 'EOS'

    word2index_lang1['SOS'] = 0
    word2index_lang1['EOS'] = 1

    word2index_lang2['SOS'] = 0
    word2index_lang2['EOS'] = 1

    index_1 = 2
    index_2 = 2

    for sentence_pair in tqdm(sentence_pairs):
        for word in sentence_pair[0].split(' '):
            if word in word2index_lang1:
                pass
            else:
                word2index_lang1[word] = index_1
                index_1 += 1
                index2word_lang1[index_1] = word

        for word in sentence_pair[1].split(' '):
            if word in word2index_lang2:
                pass
            else:
                word2index_lang2[word] = index_2
                index_2 += 1
                index2word_lang2[index_2] = word
    print(len(word2index_lang1), len(index2word_lang1))
    print(len(word2index_lang2), len(index2word_lang2))

    return word2index_lang1, word2index_lang2, index2word_lang1, index2word_lang2




sentence_pairs = read_lang_data(['commoncrawl.de-en-de.txt', 'europarl-v7_de-en-de.txt', 'news-commentary-v10_de-en-de.txt'], \
    ['commoncrawl.de-en-en.txt', 'europarl-v7_de-en-en.txt', 'news-commentary-v10_de-en-en.txt'])
sentence_dict = {'data': sentence_pairs}
pickle.dump(sentence_dict, open('paired_sentences.pkl', 'wb'))
# sentences = pickle.load(open('paired_sentences.pkl', 'rb'))
# sentence_pairs = sentences['data']
word2index_lang1, word2index_lang2, index2word_lang1, index2word_lang2 = get_lang_dict(sentence_pairs)

word2index_dict = {'lang1':word2index_lang1, 'lang2': word2index_lang2}
index2word_dict = {'lang1': index2word_lang1, 'lang2': index2word_lang2}
# sentence_dict = {'data': sentence_pairs}

pickle.dump(word2index_lang1, open('word2index.pkl','wb'))
pickle.dump(index2word_dict, open('index2word.pkl', 'wb'))
