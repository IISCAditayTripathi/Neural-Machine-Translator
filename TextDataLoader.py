import torch

from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import os
import unicodedata
import re
import time

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


def read_lang_data(lang1, lang2, root='lang_data/wmt15-de-en/'):
    # root = 'lang_data/wmt15-de-en/e'
    lang1_data = []
    for data_file in lang1:
        data = open(os.path.join(root,data_file), encoding='utf-8').read().strip().split('\n')
        lang1_data.extend(data)

    lang2_data = []
    for data_file in lang2:
        data = open(os.path.join(root,data_file), encoding='utf-8').read().strip().split('\n')
        lang2_data.extend(data)

    pairs = [[normalizeString(l1) ,normalizeString(l2)] for l1, l2 in zip(lang1_data, lang2_data)]

    return pairs

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

	for sentence_pair in sentence_pairs:
		for word in sentence_pair[0]:
			if word in word2index_lang1:
				pass
			else:
				word2index_lang1[word] = index_1
				index_1 += 1
				index2word_lang1[index_1] = word

		for word in sentence_pair[1]:
			if word in word2index_lang2:
				pass
			else:
				word2index_lang2[word] = index_2
				index_2 += 1
				index2word_lang2[index_2] = word

	return word2index_lang1, word2index_lang2, index2word_lang1, index2word_lang2




class TextDataset(Dataset):
	"""docstring for TextLoader"""
	def __init__(self, sentence_pairs, word2index_dict, index2word_dict):
		super(TextDataset, self).__init__()
		self.sentence_pairs = sentence_pairs
		self.word2index_dict = word2index_dict
		self.index2word_dict = index2word_dict
		# self.data_array = data_array
		# self.root_dir = root_dir
		# self.sentence_pairs = read_lang_data(self.data_array[0], self.data_array[1], self.root_dir)
		# self.word2index_lang1, self.word2index_lang2, self.index2word_lang1, self.index2word_lang2 = get_dict(self.sentence_pairs)

	def __len__(self):
		return len(self.sentence_pairs)

	def __getitem__(self, idx):
		sentences = self.sentence_pairs[idx]
		# print(sentences)
		lang1_array = []
#		for word in sentences[0].split(' '):
#			print(len(self.word2index_dict))
#			lang1_array.append(self.word2index_dict['lang1'][word])
		for word in sentences[0].split(' '):
			if self.word2index_dict['lang1'][word]:
				lang1_array.append(self.word2index_dict['lang1'][word])
			else:
				pass

		lang2_array = []

		for word in sentences[1].split(' '):
			if self.word2index_dict['lang2'][word]:
				lang2_array.append(self.word2index_dict['lang2'][word])
			else:
				pass

		# lang1_array = [self.word2index_dict['lang1'][word] for word in sentences[0].split(' ')]

		# lang2_array = [self.word2index_dict['lang2'][word] for word in sentences[1].split(' ')]

		lang1_array.append(self.word2index_dict['lang1']['EOS'])
		lang2_array.append(self.word2index_dict['lang2']['EOS'])
#		print(lang2_array)
		# print(len(lang1_array))
		return lang1_array, lang2_array

def _collate_fn(batch):

	def func(p):
		return len(p[0])
	def func2(p):
		return len(p[1])
#	print(lang1)
	lang1 = sorted(batch, key=lambda sample: len(sample[0]), reverse=True)
	lang1_lengths = [len(i[0]) for i in lang1]

	longest_sample_lang1 = max(lang1, key=func)[0]
	lang2 = sorted(batch, key=lambda sample: len(sample[1]), reverse=True)
	lang2_lengths = [len(i[1]) for i in lang2]
	longest_sample_lang2 = max(lang2, key=func2)[1]    
	
	minibatch_size = len(batch)
	max_seqlength_lang1 = len(longest_sample_lang1)
	max_seqlength_lang2 = len(longest_sample_lang2)
	num_classes_lang1 = 28656 
	num_classes_lang2 = 29535
	'''
	inputs_lang1 = torch.zeros(minibatch_size, max_seqlength_lang1, num_classes_lang1, dtype=torch.long)
	inputs_lang2 = torch.zeros(minibatch_size, max_seqlength_lang2, num_classes_lang2, dtype=torch.long)
	
	for x in range(minibatch_size):
		sample_lang1 = batch[x][0]
		sample_lang2 = batch[x][1]
		for i in range(len(sample_lang1)):
			inputs_lang1[x][i][sample_lang1[i]] = 1
		for i in range(len(sample_lang2)):
			inputs_lang2[x][i][sample_lang2[i]] = 1

	return inputs_lang1, inputs_lang2, lang1_lengths, lang2_lengths
	'''
			# inputs_lang1[x].narrow(1, sample_lang1[i], sample_lang1[i]).copy_(1)
	
		# [inputs_lang1[x][i][sample_lang1[i]]=1 for i in range(len(sample_lang1))]

		# [inputs_lang1[x][i][sample_lang1[i]]=1 for i in range(len(sample_lang1))]
		# print(i)
		# print(inputs_lang1[1][i][sample_lang1[i]])
	

	# '''
	inputs_lang1 = torch.zeros(minibatch_size, max_seqlength_lang1, dtype=torch.long)
	inputs_lang2 = torch.zeros(minibatch_size, max_seqlength_lang2, dtype=torch.long)


	for x in range(minibatch_size):
		sample_lang1 = batch[x][0]
		sample_lang2 = batch[x][1]

		seq_length_lang1 = len(sample_lang1)
		seq_length_lang2 = len(sample_lang2)

		sample_lang1 = torch.tensor(sample_lang1, dtype= torch.long)
		sample_lang2 = torch.tensor(sample_lang2, dtype = torch.long)
#		print(sample_lang1.shape)
		sample_lang1 = sample_lang1.view(sample_lang1.shape[0])
		sample_lang2 = sample_lang2.view(sample_lang2.shape[0])

		inputs_lang1[x].narrow(0, 0, seq_length_lang1).copy_(sample_lang1)
		inputs_lang2[x].narrow(0, 0, seq_length_lang2).copy_(sample_lang2)


	
# 	labels_one_hot_lang1 = torch.LongTensor(minibatch_size, max_seqlength_lang1, num_classes_lang1).zero_()
# #	print(labels_one_hot_lang1.shape)
# #	print(inputs_lang1.shape)
# 	labels_one_hot_lang1.scatter_(2, inputs_lang1, 1.0)

# 	labels_one_hot_lang2 = torch.LongTensor(minibatch_size, max_seqlength_lang2, num_classes_lang2).zero_()
# 	labels_one_hot_lang2.scatter_(2, inputs_lang2, 1.0)
	# 

	return inputs_lang1, inputs_lang2, lang1_lengths, lang2_lengths
	# '''


class TextLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(TextLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
