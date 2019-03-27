from __future__ import print_function, division
# import torch
from TextDataLoader import TextLoader, TextDataset
import os
import torch
# from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import pickle
from tqdm import tqdm
import argparse
from torch.nn import functional as F
from NMT import NMT
# from tensorboardX import SummaryWriter
# import console
from new_NMT import EncoderRNN, AttenDecoderRNN
from masked_cross_entropy import *

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--shuffle_data', type=bool, default=True)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--embed_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--nb_epochs', type=int, default=2)
parser.add_argument('--clip_grad', type=float, default=50.0)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--attention', type=str, default='dot')
parser.add_argument('--decoder_learning_ratio', type=float, default=5.0)

os.environ["CUDA_VISIBLE_DEVICES"]="0"

args = parser.parse_args()


sentences = pickle.load(open('paired_sentences_v5.pkl', 'rb'))
sentence_pairs = sentences['data']

train_sentence_pairs = sentence_pairs[0:10]
valid_sentence_pairs = sentence_pairs[170000:180000]

word2index_dict = pickle.load(open('word2index_v5.pkl','rb'))
index2word_dict = pickle.load(open('index2word_v5.pkl', 'rb'))


lang1_size = len(word2index_dict['lang1'])
lang2_size = len(word2index_dict['lang2'])

# index2word_lang2 = index2word['lang2']

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def train():
	text_dataset_train = TextDataset(train_sentence_pairs, word2index_dict, index2word_dict)
	data_loader = TextLoader(text_dataset_train, batch_size=args.batch_size, shuffle=args.shuffle_data, num_workers=args.num_workers)

	text_dataset_valid = TextDataset(valid_sentence_pairs, word2index_dict, index2word_dict)
	data_loader_valid = TextLoader(text_dataset_valid, batch_size=args.batch_size, shuffle=args.shuffle_data, num_workers=args.num_workers)

	train_encoder = EncoderRNN(hidden_size=args.hidden_size, source_vocab_size=lang1_size)
	train_decoder = AttenDecoderRNN(target_vocab_size=lang2_size, hidden_size=args.hidden_size, attention_model=args.attention)

	train_encoder = train_encoder.to(device)
	train_decoder = train_decoder.to(device)

	encoder_optimizer = torch.optim.Adam(train_encoder.parameters(), lr=args.lr)

	decoder_optimizer = torch.optim.Adam(train_decoder.parameters(), lr=args.lr * args.decoder_learning_ratio)
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()
	weight = torch.ones(lang2_size)
	weight[0] = 0.0
	criterion = nn.CrossEntropyLoss(weight=weight)
	criterion.to(device)

	loss = 0

	for epoch in range(args.nb_epochs):
		for phase in ['train', 'valid']:
			if phase == 'train':
				train_encoder.train()
				train_decoder.train()
			else:
				train_encoder.eval()
				train_decoder.eval()

			if phase == 'train':
				loader = tqdm(data_loader, total=len(data_loader), unit="batches")
				running_loss = 0

				for i_batch, data in enumerate(loader):
					inputs_lang1 = data[0].to(device) 
					inputs_lang2 = data[1].to(device)
					lang1_lengths = torch.tensor(data[2]).to(device)
					lang2_lengths = torch.tensor(data[3]).to(device)

					loss = train_step(inputs_lang1, lang1_lengths, inputs_lang2, lang2_lengths, train_encoder, train_decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=data[3][0])
					running_loss += loss
					loader.set_postfix(Loss=running_loss/(i_batch+1))
					# loader.set_description('{}/{}'.format(epoch, args.nb_epochs))
					# loader.update()
	

				# loader.set_postfix('{}/{} Loss:{}'.format(epoch, args.nb_epochs, running_loss/(i_batch+1)))
def masked_loss(all_decoder_outputs, input_batch_tgt):

	tgt_words_mask = (input_batch_tgt != 0).float()
	all_decoder_outputs = all_decoder_outputs.transpose(1,0)
	# print(all_decoder_outputs.shape)
	# print(input_batch_tgt.shape)

	tgt_words_log_prob = F.log_softmax(all_decoder_outputs, dim=-1)
	input_batch_tgt =  input_batch_tgt.unsqueeze(2)

	loss = torch.gather(tgt_words_log_prob, index=input_batch_tgt, dim=-1).squeeze(-1) * tgt_words_mask

	return loss.sum()



def train_step(input_batch_src, source_sent_len, input_batch_tgt, target_sent_len, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=50):
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()
	loss = 0

	encoder_outputs, encoder_hidden = encoder(input_batch_src, source_sent_len, None)

	decoder_input = Variable(torch.LongTensor([word2index_dict['lang2']['SOS']] * args.batch_size))

	decoder_hidden = encoder_hidden[:decoder.n_layers]

	max_target_length = target_sent_len[0]

	all_decoder_outputs = Variable(torch.zeros(max_target_length, args.batch_size, decoder.target_vocab_size))

	decoder_input = decoder_input.to(device)
	all_decoder_outputs = all_decoder_outputs.to(device)
	input_batch_tgt = input_batch_tgt.permute(1,0)
	i = 0
	# for t in torch.arange(max_target_length):
	for target_words in  input_batch_tgt.split(split_size=1):
		decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs)

		all_decoder_outputs[i] = decoder_output
		i += 1
		# decoder_input = input_batch_tgt[:,t]
		decoder_input = target_words.permute(1,0)

	# loss = masked_cross_entropy(all_decoder_outputs.transpose(0, 1).contiguous(), input_batch_tgt.transpose(0, 1).contiguous(), target_sent_len)

	all_decoder_outputs = all_decoder_outputs.view(-1, lang2_size)
	input_batch_tgt = input_batch_tgt.contiguous().view(-1)

	loss = criterion(all_decoder_outputs, input_batch_tgt)
	# print("here1")
	loss.backward()
	# print("here2")

	ec = torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.clip_grad)

	dc = torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.clip_grad)
	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.item()


def validation(valid_loader):
	# encoder.eval()
	# decoder.eval()

	running_loss = 0
	running_num_words = 0

	with torch.no_grad():
		loader = tqdm(valid_loader, total=len(valid_loader), unit="batches")

		for i_batch, data in enumerate(loader):
			inputs_lang1 = data[0].to(device)
			inputs_lang2 = data[0].to(device)

			lang1_lengths = torch.tensor(data[2]).to(device)
			lang2_lengths = torch.tensor(data[3]).to(device)

			encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
			decoder_input = Variable(torch.LongTensor([word2index_dict['lang2']['SOS']] * batch_size))
			decoder_hidden = encoder_hidden[:decoder.n_layers]

			max_target_length = data[3][0]

			all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))
			decoder_input = decoder_input.to(device)
			all_decoder_outputs = all_decoder_outputs.to(device)

			for t in range(max_target_length):
				decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs)
				all_decoder_outputs[t] = decoder_output
				decoder_input = input_batch_tgt[t]

			loss = masked_cross_entropy(all_decoder_outputs.transpose(0, 1).contiguous(), input_batch_tgt.transpose(0, 1).contiguous(), target_sent_len)

			running_loss += loss

			loader.set_postfix(loss/i_batch)





def evaluate(input_seq, input_lengths, max_length=50):
      
        
    # Set to not-training mode to disable dropout
    # encoder.train(False)
    # decoder.train(False)

    # encoder.eval()
    # decoder.eval()
    
    encoder_outputs, encoder_hidden = encoder(input_seq, input_lengths, None)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([word2index_dict['lang2']['SOS']]), volatile=True) # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
    
    decoder_input = decoder_input.to(device)

    # Store output words and attention states
    decoded_words = []
    decoder_attentions = torch.zeros(max_length + 1, max_length + 1)
    OS_token = 2
    
    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word_dict['lang2'][ni])
            
        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([ni]))

        decoder_input = decoder_input.to(device)

    # Set back to training mode
    # encoder.train()
    # decoder.train()
    
    return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]



if args.mode=='train':
	train()

if args.mode=='eval':
	evaluate()