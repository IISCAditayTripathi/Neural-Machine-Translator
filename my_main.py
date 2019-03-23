from __future__ import print_function, division
# import torch
from TextDataLoader import TextLoader, TextDataset
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
import argparse
from NMT import NMT
from tensorboardX import SummaryWriter
import console

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=25)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--shuffle_data', type=bool, default=True)
parser.add_argument('--num_workers', type=int, default=25)
parser.add_argument('--embed_size', type=int, default=256)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--nb_epochs', type=int, default=10)
parser.add_argument('--clip_grad', type=float, default=3.0)
parser.add_argument('--mode', type=str, default='train')



args = parser.parse_args()

import warnings
warnings.filterwarnings("ignore")
data_array = ['commoncrawl.de-en-de.txt', 'europarl-v7_de-en-de.txt', 'news-commentary-v10_de-en-de.txt'], \
    ['commoncrawl.de-en-en.txt', 'europarl-v7_de-en-en.txt', 'news-commentary-v10_de-en-en.txt']
root = 'lang_data/wmt15-de-en/'

sentences = pickle.load(open('paired_sentences_v4_1.2.pkl', 'rb'))
sentence_pairs = sentences['data']
# print(len(sentence_pairs))
# aditay

train_sentence_pairs = sentence_pairs[0:400000]
valid_sentence_pairs = sentence_pairs[400000:]

word2index_dict = pickle.load(open('word2index_v4_1.pkl','rb'))
index2word_dict = pickle.load(open('index2word_v4_1.pkl', 'rb'))


lang1_size = len(word2index_dict['lang1'])
lang2_size = len(word2index_dict['lang2'])

# index2word_lang2 = index2word['lang2']

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


# print(lang1_size, lang2_size)






def validation_loss(model, valid_loader, batch_size=10):
	was_training = model.training
	model.eval()

	running_loss = 0
	running_num_words = 0

	with torch.no_grad():
		loader = tqdm(valid_loader, total=len(valid_loader), unit="batches")
		for i_batch, data in enumerate(loader):
			try:
				inputs_lang1 = data[0].to(device)
				inputs_lang2 = data[0].to(device)

				lang1_lengths = torch.tensor(data[2]).to(device)
				lang2_lengths = torch.tensor(data[3]).to(device)
		
				loss = -model(inputs_lang1, inputs_lang2, lang1_lengths, lang2_lengths).sum()

				running_loss += loss.item()
				target_word_num_to_predict = sum(len(i[1:]) for i in inputs_lang2)

				running_num_words += target_word_num_to_predict
				loader.set_postfix(ppl=(np.exp(running_loss/running_num_words)))
			except:
				pass
		ppl = running_loss/running_num_words
		print("validation Perplexity: %f"%(ppl))
		return ppl

def translate():
	print('Strting translating')

	model = NMT(target_dict=word2index_dict['lang2'], index2word=index2word_dict['lang2'], embedding_size=args.embed_size, hidden_size=args.hidden_size, src_vocab_size=lang1_size, target_vocab_size=lang2_size, device=device)
	model.load_state_dict(torch.load('checkpoints/NMT_1_0.07225268958970377_.pt'))

	model = model.to(device)

	text_dataset_valid = TextDataset(valid_sentence_pairs, word2index_dict, index2word_dict)

	data_loader_valid = TextLoader(text_dataset_valid, batch_size=1, shuffle=args.shuffle_data, num_workers=args.num_workers)
	file = open('decoded_text.txt', 'w')
	with torch.no_grad():
		model.eval()
		for data in data_loader_valid:
			inputs_lang1 = data[0].to(device) 
			inputs_lang2 = data[1].to(device)
			lang1_lengths = torch.tensor(data[2]).to(device)
			lang2_lengths = torch.tensor(data[3]).to(device)

			decoded_sentence = model.beam_search(inputs_lang1)
			# print(decoded_sentence)
			for word in decoded_sentence[0][0]:
				file.write(word)
				file.write(' ')

			file.write('\n')
			# aditay


def train():
	print('Starting training ###########')
	# console.log('##')

	text_dataset_train = TextDataset(train_sentence_pairs, word2index_dict, index2word_dict)

	data_loader = TextLoader(text_dataset_train, batch_size=args.batch_size, shuffle=args.shuffle_data, num_workers=args.num_workers)

	text_dataset_valid = TextDataset(valid_sentence_pairs, word2index_dict, index2word_dict)

	data_loader_valid = TextLoader(text_dataset_valid, batch_size=args.batch_size, shuffle=args.shuffle_data, num_workers=args.num_workers)


	model = NMT(target_dict=word2index_dict['lang2'], index2word=index2word_dict['lang2'], embedding_size=args.embed_size, hidden_size=args.hidden_size, src_vocab_size=lang1_size, target_vocab_size=lang2_size, device=device)
	model = model.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


	num_iterations = 0
	writer = SummaryWriter()
	# writer.add_graph(model, input_to_model=(torch.autograd.Variable(torch.Tensor(1,20, dtype=torch.long).to(device)), torch.autograd.Variable(torch.Tensor(1,20, dtype=torch.long).to(device)), torch.autograd.Variable(torch.LongTensor(20).to(device)), torch.autograd.Variable(torch.LongTensor(20).to(device))) )
	best_loss = float('inf')
	for epoch in range(args.nb_epochs):

		for phase in ['train', 'valid']:
			if phase == 'train':
				model.train()
			else:
				model.eval() 
			if phase == 'train':
				train_epochs = 0
				loader = tqdm(data_loader, total=len(data_loader), unit="batches")
				batch_loss = 0

				for i_batch, data in enumerate(loader):
					# num_iterations += 1
					# train_epochs += 1
		
					inputs_lang1 = data[0].to(device) 
					inputs_lang2 = data[1].to(device)
					lang1_lengths = torch.tensor(data[2]).to(device)
					lang2_lengths = torch.tensor(data[3]).to(device)

					optimizer.zero_grad()
					# loss = float('inf')
					# loader.set_postfix(loss=batch_loss.cpu().detach().numpy()/i_batch, size=inputs_lang1.shape)

					loss = - model(inputs_lang1, inputs_lang2, lang1_lengths, lang2_lengths)

					loss = loss.sum()/args.batch_size
					if i_batch %1000 == 0:
						writer.add_scalar('loss/gender_loss', loss, i_batch)
						for name, param in model.named_parameters():
							writer.add_histogram(name, param.clone().cpu().data.numpy(), i_batch)
					# print(loss)

					loss.backward()
					grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)

					optimizer.step()

					batch_loss += loss.sum()
	
					loader.set_postfix(loss=batch_loss.cpu().detach().numpy()/i_batch)

			if phase == 'valid':

				ppl = validation_loss(model, data_loader_valid)
				if best_loss > (ppl):
					best_loss = ppl
					torch.save(model.state_dict(), 'checkpoints/NMT_'+str(epoch)+'_'+str(ppl)+'_'+'.pt')


if args.mode=='train':
	train()
if args.mode=='test':
	translate()
