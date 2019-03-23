import pickle
import operator
from collections import defaultdict
from tqdm import tqdm


word_counts = pickle.load(open('word_count_v4.pkl', 'rb'))
word_counts_lang1 = word_counts['lang1']
sorted_word_counts_lang1 = sorted(word_counts_lang1.items(), key=operator.itemgetter(1))

word_counts_lang2 = word_counts['lang2']

sorted_word_counts_lang2 = sorted(word_counts_lang2.items(), key=operator.itemgetter(1))

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


min_count = 5

index = 2

for key, count in tqdm(sorted_word_counts_lang1):
	if count > min_count:
		word2index_lang1[key] = index
		index2word_lang1[index] = key
		index += 1


index = 2
for key, count in tqdm(sorted_word_counts_lang2):
	if count > min_count:
		word2index_lang2[key] = index
		index2word_lang2[index] = key
		index += 1

print(len(word2index_lang1), len(index2word_lang1))
print(len(word2index_lang2), len(index2word_lang2))

aditay

word2index_dict = {'lang1':word2index_lang1, 'lang2': word2index_lang2}
index2word_dict = {'lang1': index2word_lang1, 'lang2': index2word_lang2}
# word_count = {'lang1': word_count_lang1, 'lang2': word_count_lang2}
# sentence_dict = {'data': sentence_pairs}

pickle.dump(word2index_dict, open('word2index_v3.pkl','wb'))
pickle.dump(index2word_dict, open('index2word_v3.pkl', 'wb'))
# pickle.dump(word_count, open('word_count.pkl', 'wb'))



