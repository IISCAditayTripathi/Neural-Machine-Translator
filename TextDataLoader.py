import torch

from torch.utils.data import Dataset
from collections import defaultdict

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

    return language_1, language_2, pairs

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




class TextLoader(Dataset):
	"""docstring for TextLoader"""
	def __init__(self, data_array, root_dir):
		super(TextLoader, self).__init__()
		self.data_array = data_array
		self.root_dir = root_dir
		self.sentence_pairs = read_lang_data(self.data_array[0], self.data_array[1], self.root_dir)
		self.word2index_lang1, self.word2index_lang2, self.index2word_lang1, self.index2word_lang2 = get_dict(self.sentence_pairs)

	def __len__(self):
		return len(self.sentence_pairs)

	def __getitem__(self, idx):
		sentences = self.sentence_pairs[idx]

		lang1_array = [self.word2index_lang1[word] for word in sentences[0]]

		lang2_array = [self.word2index_lang2[word] for word in sentences[1]]

		lang1_array.append(self.word2index_lang1['EOS'])
		lang2_array.append(self.word2index_lang2['EOS'])

		return lang1_array, lang2_array


		
