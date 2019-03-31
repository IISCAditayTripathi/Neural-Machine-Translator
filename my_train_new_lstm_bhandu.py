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
# from tensorboardX import SummaryWriter
# import console
from new_NMT_lstm import EncoderRNN, AttenDecoderRNN, BahdanauDecoder

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--shuffle_data', type=bool, default=True)
parser.add_argument('--num_workers', type=int, default=25)
parser.add_argument('--embed_size', type=int, default=256)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--nb_epochs', type=int, default=50)
parser.add_argument('--clip_grad', type=float, default=5.0)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--attention', type=str, default='dot')
parser.add_argument('--decoder_learning_ratio', type=float, default=5.0)
parser.add_argument('--gpu', type=int, default=0)

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

args = parser.parse_args()


sentences = pickle.load(open('paired_sentences_v5_50.pkl', 'rb'))
sentence_pairs = sentences['data']

train_sentence_pairs = sentence_pairs[0:144000]
valid_sentence_pairs = sentence_pairs[144000:162000]

test_sentence_pairs = sentence_pairs[162000:]


word2index_dict = pickle.load(open('word2index_v5_50.pkl','rb'))
index2word_dict = pickle.load(open('index2word_v5_50.pkl', 'rb'))


lang1_size = len(word2index_dict['lang1'])
lang2_size = len(word2index_dict['lang2'])

# index2word_lang2 = index2word['lang2']

use_cuda = torch.cuda.is_available()
device = torch.device(args.gpu if use_cuda else "cpu")


def train():
	text_dataset_train = TextDataset(train_sentence_pairs, word2index_dict, index2word_dict)
	data_loader = TextLoader(text_dataset_train, batch_size=args.batch_size, shuffle=args.shuffle_data, num_workers=args.num_workers)

	text_dataset_valid = TextDataset(valid_sentence_pairs, word2index_dict, index2word_dict)
	data_loader_valid = TextLoader(text_dataset_valid, batch_size=1, shuffle=args.shuffle_data, num_workers=args.num_workers)

	train_encoder = EncoderRNN(hidden_size=args.hidden_size, source_vocab_size=lang1_size)
	train_decoder = BahdanauDecoder(target_vocab_size=lang2_size, hidden_size=args.hidden_size, attention_model=args.attention)

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
	i = 0
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
				decoder_init_input = Variable(torch.LongTensor([word2index_dict['lang2']['SOS']] * args.batch_size)).to(device)
				for i_batch, data in enumerate(loader):
					i += 1
					inputs_lang1 = data[0].to(device) 
					inputs_lang2 = data[1].to(device)
					lang1_lengths = torch.tensor(data[2]).to(device)
					lang2_lengths = torch.tensor(data[3]).to(device)

					loss = train_step(inputs_lang1, lang1_lengths, inputs_lang2, lang2_lengths, train_encoder, train_decoder, encoder_optimizer, decoder_optimizer, criterion, decoder_init_input, max_length=data[3][0])
					running_loss += loss
#					if (i %10) == 0:
#						print(i_batch, running_loss/i_batch)
					loader.set_postfix(Loss=running_loss/(i_batch+1))
					# loader.set_description('{}/{}'.format(epoch, args.nb_epochs))
					# loader.update()
				torch.save(train_encoder.state_dict(), 'checkpoints/new_checkpoint_50_LSTM/NMT_encoder_bhanudu_he'+str(args.attention)+str(epoch)+'_'+str(running_loss/i_batch)+'.pt')
				torch.save(train_decoder.state_dict(), 'checkpoints/new_checkpoint_50_LSTM/NMT_decoder_bhanudu_he'+str(args.attention)+str(epoch)+'_'+str(running_loss/i_batch)+'.pt')

			if phase == 'valid':
				validation()
				


				# loader.set_postfix('{}/{} Loss:{}'.format(epoch, args.nb_epochs, running_loss/(i_batch+1)))
def masked_loss(all_decoder_outputs, input_batch_tgt):

	tgt_words_mask = (input_batch_tgt != 0).float()
	all_decoder_outputs = all_decoder_outputs.transpose(1,0)
	tgt_words_log_prob = F.log_softmax(all_decoder_outputs, dim=-1)
	input_batch_tgt =  input_batch_tgt.unsqueeze(2)

	loss = torch.gather(tgt_words_log_prob, index=input_batch_tgt, dim=-1).squeeze(-1) * tgt_words_mask

	return loss.sum()



def train_step(input_batch_src, source_sent_len, input_batch_tgt, target_sent_len, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, decoder_init_input, max_length=50):
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()
	loss = 0

	encoder_outputs, encoder_hidden = encoder(input_batch_src, source_sent_len, None)

#	decoder_input = Variable(torch.LongTensor([word2index_dict['lang2']['SOS']] * args.batch_size))
	decoder_input = decoder_init_input
	# cell_input = decoder_init_input
	decoder_hidden = encoder_hidden[:decoder.n_layers]
	decoder_hidden = decoder_hidden.view(args.batch_size, args.hidden_size)

	max_target_length = target_sent_len[0]

#	all_decoder_outputs = Variable(torch.zeros(max_target_length, args.batch_size, decoder.target_vocab_size))

#	decoder_input = decoder_input.to(device)

#	all_decoder_outputs = all_decoder_outputs.to(device)
	input_batch_tgt = input_batch_tgt.permute(1,0)
	i = 0
	all_decoder_outputs = []
	# print("aditay",decoder_hidden.shape)
	cell_input = decoder_hidden
	# for t in torch.arange(max_target_length):
	for target_words in  input_batch_tgt.split(split_size=1):
		# print(decoder_input.shape)
		decoder_output, decoder_hidden, cell_input, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs, cell_input)

#		all_decoder_outputs[i] = decoder_output
#		i += 1
		# decoder_input = decoder_input.
		all_decoder_outputs.append(decoder_output)
		# decoder_input = input_batch_tgt[:,t]
		# print(target_words.shape)
		decoder_input = target_words.permute(1,0).squeeze(1)
		# print(decoder_input.shape)

	# loss = masked_cross_entropy(all_decoder_outputs.transpose(0, 1).contiguous(), input_batch_tgt.transpose(0, 1).contiguous(), target_sent_len)
	all_decoder_outputs = torch.stack(all_decoder_outputs)
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


def evaluate(max_length=50):
      
        
    # Set to not-training mode to disable dropout
    # encoder.train(False)
    # decoder.train(False)

    # encoder.eval()
    # decoder.eval()
	device = torch.device('cuda:0')
	encoder = EncoderRNN(hidden_size=args.hidden_size, source_vocab_size=lang1_size)
	decoder = BahdanauDecoder(target_vocab_size=lang2_size, hidden_size=args.hidden_size, attention_model=args.attention)
	encoder = encoder.to(device)
	decoder = decoder.to(device)
	
	encoder.load_state_dict(torch.load('/scratche/home/aditay/NLU_assignment2/checkpoints/new_checkpoint_50_LSTM/NMT_encoder_bhanududot49_2.390987828460857.pt', map_location='cuda:0'))
	decoder.load_state_dict(torch.load('/scratche/home/aditay/NLU_assignment2/checkpoints/new_checkpoint_50_LSTM/NMT_decoder_bhanududot49_2.390987828460857.pt', map_location='cuda:0'))
	encoder.eval()
	decoder.eval()

	text_dataset_valid = TextDataset(test_sentence_pairs, word2index_dict, index2word_dict)
	data_loader_valid = TextLoader(text_dataset_valid, batch_size=1, shuffle=args.shuffle_data, num_workers=args.num_workers)
	loader = tqdm(data_loader_valid, total=len(data_loader_valid), unit="batches")
	original_english_text = open('original_bandu_english_text.txt', 'w')
	original_german_text = open('original_bandu_german_text.txt', 'w')
	predicted_english = open('predicted_bandu_english_text.txt', 'w')
	# decoder_input = Variable(torch.LongTensor([word2index_dict['lang2']['SOS']] * args.batch_size)).to(device)

	for i_batch, data in enumerate(loader):
		inputs_lang1 = data[0].to(device) 
		inputs_lang2 = data[1].to(device)
		lang1_lengths = torch.tensor(data[2]).to(device)
		lang2_lengths = torch.tensor(data[3]).to(device)

		encoder_outputs, encoder_hidden = encoder(inputs_lang1, lang1_lengths)

		decoder_input = Variable(torch.LongTensor([word2index_dict['lang2']['SOS']]*1), volatile=True)

		decoder_hidden = encoder_hidden[:decoder.n_layers].squeeze(0)
		decoder_input = decoder_input.to(device)
		cell_input = decoder_hidden

		decoded_words = []
		decoder_attentions = torch.zeros(max_length + 1, max_length + 1)
		for i in inputs_lang1.permute(1,0).split(split_size=1):
			# print(i)
			# print(index2word_dict['lang1'][i.item()])
			original_german_text.write(str(index2word_dict['lang1'][i.item()]))
			original_german_text.write(' ')
		original_german_text.write('\n')

		for i in inputs_lang2.permute(1,0).split(split_size=1):
			original_english_text.write(str(index2word_dict['lang2'][i.item()]))
			original_english_text.write(' ')
		original_english_text.write('\n')

		for i in range(max_length):
			decoder_output, decoder_hidden, cell_input, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs, cell_input)


			topv, topi = decoder_output.data.topk(1)
			predicted_index = topi[0][0].item()
			# print(predicted_index.item())

			if predicted_index == 2:
				# print(index2word_dict['lang2'][predicted_index])
				predicted_english.write(str(index2word_dict['lang2'][predicted_index]))
				predicted_english.write(' ')
				# decoded_words.append('EOS')
				break
			else:
				predicted_english.write(str(index2word_dict['lang2'][predicted_index]))
				predicted_english.write(' ')
				# decoded_words.append(index2word_dict['lang2'][predicted_index][])
				# decode

			decoder_input = Variable(torch.LongTensor([predicted_index]))
			decoder_input = decoder_input.to(device)
		# print(decoded_words)
		predicted_english.write('\n')

	return decoded_words, decoder_attentions[:i+1, :len(encoder_outputs)]



if args.mode=='train':
	train()

if args.mode=='eval':
	evaluate()
