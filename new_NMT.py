import torch
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

device = torch.device('cuda:1')

class EncoderRNN(nn.Module):
	"""docstring for EncoderRNN"""
	def __init__(self, hidden_size, source_vocab_size, n_layers=1, dropout=0.1):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.source_vocab_size = source_vocab_size
		self.n_layers = n_layers
		self.dropout = dropout

		self.embedding = nn.Embedding(self.source_vocab_size, self.hidden_size) 

		self.GRU = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers, dropout=self.dropout, bidirectional=True)


	def forward(self, source_sentence, source_sentence_len, hidden=None):
		embedded_sentences = self.embedding(source_sentence).view(1, 1, -1)
		output = output.view(embedded_sentences.shape[1], embedded_sentences.shape[0], embedded_sentences.shape[2])
		packed_output = pack_padded_sequence(embedded_sentences, source_sentence_len)

		outputs, hidden = self.GRU(packed_output, hidden)

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

	def forward(self, hidden, encoder_outputs):
		max_length = encoder_outputs.size(0)

		batch_size = encoder_outputs.size(1)

		attention_weights = Variable(torch.zeros(this_batch_size, max_len))

		attention_weights = attention_weights.to(device)

		for b in len(batch_size):
			for i in range(max_len):

				attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

		return F.softmax(attn_energies).unsqueeze(1)

	def score(self, hidden, encoder_output):
		
		if self.method == 'dot':
			weight = hidden.dot(encoder_output)
			return weight

		elif self.method == 'general':

			weight = self.atten(encoder_output)
			weight = hidden.dot(weight)
			return weight

		elif self.method == 'concatenate':
			weight = self.atten(torch.concat((hidden, encoder_output), 1))
			weight = self.v.dot(weight)
			return weight

class AttenDecoderRNN(object):
	"""docstring for AttenDecoderRNN"""
	def __init__(self, attention_model, hidden_size, target_vocab_size, n_layers=1, dropout=0.1):
		super(AttenDecoderRNN, self).__init__()
		self.atten_model = atten_model
		self.hidden_size = hidden_size
		self.target_vocab_size. = target_vocab_size
		self.n_layers = n_layers
		self.dropout = dropout

		self.embedding = nn.Embedding(self.target_vocab_size, self.hidden_size)
		self.embedding_dropout = nn.Dropout(dropout=self.dropout)
		self.GRU = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers, dropout=self.dropout)

		self.concat = nn.Linear(2*self.hidden_size, self.hidden_size)
		self.out = nn.Linear(self.hidden_size, self.target_vocab_size)

		self.atten = Attention(self.atten_model, self.hidden_size)


	def forward(self, input_seq, last_hidden, encoder_outputs):
		batch_size = input_seq.size(0)

		embedded = self.embedding(input_seq)
		embedded = self.embedding_dropout(embedded)
		embedded = embedded.view(1, batch_size, self.hidden_size)

		rnn_output, hidden = self.GRU(embedded, last_hidden)

		attention_weights = self.atten(rnn_output, encoder_outputs)

		context = attention_weights.bmm(encoder_outputs.transpose(0, 1))

		rnn_output = rnn_output.squeeze(0)

		context = context.squeeze(1)

		concat_input = torch.cat((rnn_output, context), 1)

		concat_output = F.tanh(self.concat(concat_input))

		output = self.out(concat_output)

		return output, hidden, attn_weights



