import pickle
import os
from collections import defaultdict
from tqdm import tqdm
import unicodedata
import string
import re
import random

german_data = open('/scratche/home/aditay/NLU_assignment2/lang_data/wmt15-de-en/german_data_tokenized.txt', 'r').read().strip().split('\n')

english_data = open('/scratche/home/aditay/NLU_assignment2/lang_data/wmt15-de-en/english_data_tokenized.txt', 'r').read().strip().split('\n')



german_data = german_data[0:800000]
english_data = english_data[0:800000]

def normalizeString(s):
    # s = unicodeToAscii(s.lower().strip())
    s = s.lower().strip()
    # s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    # # table = string.maketrans(",")
    # s = s.translate(maketrans("","", string.punctuation))
    # s = s.translate(maketrans("","", digits))
    return s
pairs = []



for i in tqdm(zip(german_data, english_data)):
        pairs.append([normalizeString(i[0]), normalizeString(i[1])])
print(pairs[0:14])

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def get_lang_dict(sentence_pairs):
    word2index_lang1 = defaultdict(int)
    word2index_lang2 = defaultdict(int)
    index2word_lang1 = defaultdict(int)
    index2word_lang2 = defaultdict(int)
    index2word_lang1[0] = 'SOS'
    index2word_lang1[1] = 'EOS'

    index2word_lang2[0] = 'SOS'
    index2word_lang2[1] = 'EOS'

    word2index_lang1['SOS'] = 1
    word2index_lang1['EOS'] = 2

    word2index_lang2['SOS'] = 1
    word2index_lang2['EOS'] = 2

    index_1 = 3
    index_2 = 3
    word_count_lang1 = defaultdict(int)
    word_count_lang2 = defaultdict(int)

    for sentence_pair in tqdm(sentence_pairs):
        for word in sentence_pair[0].split(' '):
            if word in word2index_lang1:
                word_count_lang1[word] = word_count_lang1[word] + 1
                # pass
            else:
                word2index_lang1[word] = index_1
                index_1 += 1
                index2word_lang1[index_1] = word

                word_count_lang1[word] = 1

        for word in sentence_pair[1].split(' '):
            if word in word2index_lang2:
                word_count_lang2[word] = word_count_lang2[word] + 1
                # pass
            else:
                word2index_lang2[word] = index_2
                index_2 += 1
                index2word_lang2[index_2] = word

                word_count_lang2[word] = 1

    print(len(word2index_lang1), len(index2word_lang1))
    print(len(word2index_lang2), len(index2word_lang2))

    return word2index_lang1, word2index_lang2, index2word_lang1, index2word_lang2, word_count_lang1, word_count_lang2

sentence_dict = {'data': pairs}
pickle.dump(sentence_dict, open('paired_sentences_v4_1.pkl', 'wb'))


word2index_lang1, word2index_lang2, index2word_lang1, index2word_lang2, word_count_lang1, word_count_lang2 = get_lang_dict(pairs)

word2index_dict = {'lang1':word2index_lang1, 'lang2': word2index_lang2}
index2word_dict = {'lang1': index2word_lang1, 'lang2': index2word_lang2}
word_count = {'lang1': word_count_lang1, 'lang2': word_count_lang2}
# sentence_dict = {'data': sentence_pairs}

pickle.dump(word2index_dict, open('word2index_v4_1.pkl','wb'))
pickle.dump(index2word_dict, open('index2word_v4_1.pkl', 'wb'))
pickle.dump(word_count, open('word_count_v4_1.pkl', 'wb'))
