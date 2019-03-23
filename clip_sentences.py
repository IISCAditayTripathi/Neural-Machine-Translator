import pickle

from tqdm import tqdm


sentences = pickle.load(open('paired_sentences_v4_1.pkl', 'rb'))
sentence_pairs = sentences['data']

sentences2 = []
threshold = 150
for data in tqdm(sentence_pairs):
	if len(data[0]) > threshold:
		pass
	else:
		sentences2.append(data)

sentences2_dict = {'data': sentences2}

pickle.dump(sentences2_dict, open('paired_sentences_v4_1.2.pkl', 'wb'))

