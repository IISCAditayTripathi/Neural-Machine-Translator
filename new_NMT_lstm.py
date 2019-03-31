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
		outputs = embedded_sentences.transpose(1,0)
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

		attention_weights = self.score(hidden, encoder_outputs)
		normalized_attention = F.softmax(attention_weights).unsqueeze(1)

		return normalized_attention
	# @profile
	def score(self, hidden, encoder_output):
		
		if self.method == 'dot':
			hidden = hidden.unsqueeze(1)
			encoder_output = encoder_output.transpose(1,0).transpose(2,1)
			weight = torch.bmm(hidden, encoder_output)/self.hidden_size
			return weight.squeeze(1)

		elif self.method == 'mul':

			weight = self.atten(encoder_output)
			hidden = hidden.unsqueeze(1)
			weight = weight.transpose(1,0).transpose(2,1)
			weight = torch.bmm(hidden, weight)
			return weight.squeeze(1)

		elif self.method == 'concatenate':

			hidden = hidden.expand(encoder_output.size())
			weight = self.atten(torch.cat((hidden, encoder_output), 2))
			weight = weight.transpose(1,0).transpose(2,1)
			weight = torch.bmm(self.v.expand(weight.size(0),self.v.size(0), self.v.size(1)), weight)
			return F.softmax(weight.squeeze(1))
class BahdanauDecoder(nn.Module):
	"""docstring for BahdanauDecoder"""
	def __init__(self, attention_model, hidden_size, target_vocab_size, n_layers=1, dropout=0.1):
		super(BahdanauDecoder, self).__init__()
		self.atten_model = attention_model
		self.hidden_size = hidden_size
		self.target_vocab_size = target_vocab_size
		self.n_layers = n_layers
		# self.dropout = dropout

		self.embedding = nn.Embedding(self.target_vocab_size, self.hidden_size)
		self.dropout = nn.Dropout(dropout)
		self.attention = Attention('concatenate', self.hidden_size)
		self.lstm = nn.LSTMCell(self.hidden_size*2, self.hidden_size, bias=True)
		self.out = nn.Linear(self.hidden_size*2, self.target_vocab_size)

	def forward(self, input_seq, last_hidden, encoder_outputs, cell_input):
		embedded = self.embedding(input_seq)
		embedded = self.dropout(embedded)

		attention_weights = self.attention(last_hidden[-1], encoder_outputs)


		attention = torch.bmm(attention_weights, encoder_outputs.transpose(1,0))


		lstm_input = torch.cat((embedded, attention.squeeze(1)),1)

		hidden, cell_state = self.lstm(lstm_input, (last_hidden, cell_input))

		output = self.out(torch.cat((hidden, attention.squeeze(1)),1))
		
		return output, hidden, cell_state, attention_weights
		
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

		hidden, cell_state = self.GRU(embedded, (last_hidden, cell_input))

		attention_weights = F.softmax(self.atten(hidden, encoder_outputs))
		context = attention_weights.bmm(encoder_outputs.transpose(0, 1))

		rnn_output = hidden
		context = context.squeeze(1)

		concat_input = torch.cat((rnn_output, context), 1)

		concat_output = torch.tanh(self.concat(concat_input))

		output = self.out(concat_output)

		return output, hidden, cell_state, attention_weights



