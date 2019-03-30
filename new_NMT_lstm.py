 #!/usr/bin/env python3 -W ignore::DeprecationWarning
import torch
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

device = torch.device('cuda:1')

def init_weights(m):
	if type(m) == nn.Linear:
	    torch.nn.init.xavier_uniform(m.weight)

class EncoderRNN(nn.Module):
	"""docstring for EncoderRNN"""
	def __init__(self, hidden_size, source_vocab_size, n_layers=1, dropout=0.1):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.source_vocab_size = source_vocab_size
		self.n_layers = n_layers
		self.dropout = dropout

		self.embedding = nn.Embedding(self.source_vocab_size, self.hidden_size) 

		self.GRU = nn.LSTM(self.hidden_size, self.hidden_size, self.n_layers, dropout=self.dropout, bidirectional=True)


	def forward(self, source_sentence, source_sentence_len, hidden=None):
		embedded_sentences = self.embedding(source_sentence)
		# print(embedded_sentences.shape)
		# outputs = embedded_sentences.view(embedded_sentences.shape[1], embedded_sentences.shape[0], embedded_sentences.shape[2])
		# print(outputs.shape)
		outputs = embedded_sentences.transpose(1,0)
		# print(outputs.shape)
		packed_output = pack_padded_sequence(outputs, source_sentence_len)

		outputs, (hidden, cell_state) = self.GRU(packed_output)

		outputs, outputs_length = pad_packed_sequence(outputs)

		outputs = outputs[:,:,:self.hidden_size] + outputs[:, :, self.hidden_size:]
		return outputs, hidden
	
	def init_hidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)



class Attention(nn.Module):
	"""docstring for Attention"""
	def __init__(self, method, hidden_size):
		super(Attention, self).__init__()
		self.method = method

		self.hidden_size = hidden_size

		if self.method == 'general':
			self.atten = nn.Linear(self.hidden_size, self.hidden_size)
		elif self.method == 'concatenate':
			self.atten = nn.Linear(2*self.hidden_size, self.hidden_size)
			self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

	def get_attention_mask(self, src_encodings, src_sentence_length):
		src_sent_masks = torch.zeros(src_encodings.size(0), src_encodings.size(1), dtype=torch.float)

		for e_id, src_len in enumerate(src_sentence_length):
			src_sent_masks[e_id, src_len:] = 1


		return src_sent_masks.to(self.device)

	# @profile
	def forward(self, hidden, encoder_outputs):
		max_length = encoder_outputs.size(0)

		batch_size = encoder_outputs.size(1)

		# attention_weights = Variable(torch.zeros(batch_size, max_length))
		attention_weights = torch.ones(batch_size, max_length, device=device)

		# print(hidden.unsqueeze(1).shape)
		# print(encoder_outputs.transpose(1,0).transpose(2,1).shape)

		# attention_weights = attention_weights.to(device)
		attention_weights = self.score(hidden, encoder_outputs)

		# for b in torch.arange(batch_size):
		# 	for i in torch.arange(max_length):
		# 		# print(hidden[b,:].shape)
		# 		print(hidden[b,:].shape)
		# 		print(encoder_outputs[i,b].shape)
		# 		attention_weights[b, i] = torch.bmm(hidden[b,:], encoder_outputs[i,b].squeeze(0))
				# attention_weights[b, i] = self.score(hidden[b,:], encoder_outputs[i, b].squeeze(0))

		normalized_attention = F.softmax(attention_weights).unsqueeze(1)
		# normalized_attention = nn.Softmax(attention_weights)
		# normalized_attention = normalized_attention.unsqueeze(1)

		return normalized_attention
	# @profile
	def score(self, hidden, encoder_output):
		
		if self.method == 'dot':
			hidden = hidden.unsqueeze(1)
			encoder_output = encoder_output.transpose(1,0).transpose(2,1)
			# print(hidden.shape)
			# weight = hidden.dot(encoder_output)
			weight = torch.bmm(hidden, encoder_output)
			# print(weight.shape)
			return weight.squeeze(1)

		elif self.method == 'general':

			weight = self.atten(encoder_output)
			weight = hidden.dot(weight)
			return weight

		elif self.method == 'concatenate':
			weight = self.atten(torch.concat((hidden, encoder_output), 1))
			weight = self.v.dot(weight)
			return weight

class AttenDecoderRNN(nn.Module):
	"""docstring for AttenDecoderRNN"""
	def __init__(self, attention_model, hidden_size, target_vocab_size, n_layers=1, dropout=0.1):
		super(AttenDecoderRNN, self).__init__()
		self.atten_model = attention_model
		self.hidden_size = hidden_size
		self.target_vocab_size = target_vocab_size
		self.n_layers = n_layers
		self.dropout = dropout

		self.embedding = nn.Embedding(self.target_vocab_size, self.hidden_size)
		self.embedding_dropout = nn.Dropout(self.dropout)
		# self.GRU = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers, dropout=self.dropout)
		self.GRU = nn.LSTMCell(self.hidden_size, self.hidden_size, bias=True)
		self.concat = nn.Linear(2*self.hidden_size, self.hidden_size)
		self.out = nn.Linear(self.hidden_size, self.target_vocab_size)

		self.atten = Attention(self.atten_model, self.hidden_size)


	def forward(self, input_seq, last_hidden, encoder_outputs, cell_input):
		batch_size = input_seq.size(0)

		embedded = self.embedding(input_seq)
		embedded = self.embedding_dropout(embedded)
		# print(embedded.shape)
		# aditay
		# embedded = embedded.unsqueeze(0)
		# print(embedded.shape)
		# print(last_hidden.shape)
		# print(cell_input.shape)
		# embedded = embedded.view(1, batch_size, self.hidden_size)
		# print(embedded.shape)
		# embedded = embedded.view(1, batch_size, self.hidden_size)
		# print(last_hidden.shape)
		# print(cell_input.shape)
		# print(embedded.shape)
		hidden, cell_state = self.GRU(embedded, (last_hidden, cell_input))
		# print(hidden.shape)
		# print(hidden.shape)
		# print(encoder_outputs.shape)

		attention_weights = self.atten(hidden, encoder_outputs)
		# print(aditay)
		# print(encoder_outputs.shape)
		context = attention_weights.bmm(encoder_outputs.transpose(0, 1))

		rnn_output = hidden
		# print(context.shape)
		context = context.squeeze(1)
		# print(rnn_output.shape)
		# print(context.shape)

		concat_input = torch.cat((rnn_output, context), 1)

		concat_output = torch.tanh(self.concat(concat_input))

		output = self.out(concat_output)
		# print(output.shape)

		return output, hidden, cell_state, attention_weights



