import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import math
import pickle
import sys
import time
from collections import namedtuple

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm


def init_weights(m):
	if type(m) == nn.Linear:
	    torch.nn.init.xavier_uniform(m.weight)
    # m.bias.data.fill_(0.01)

class NMT(nn.Module):
	"""docstring for NMT"""
	def __init__(self, embedding_size, hidden_size, src_vocab_size, target_vocab_size,device, dropout_rate=0.2, feed_input = True ):
		super(NMT, self).__init__()
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.dropout_rate = dropout_rate
		self.feed_input = feed_input
		self.src_vocab_size = src_vocab_size
		self.target_vocab_size = target_vocab_size
		self.device = device

		self.src_embedding = nn.Embedding(self.src_vocab_size, self.embedding_size, padding_idx = 0)
		self.target_embedding = nn.Embedding(self.target_vocab_size, self.embedding_size, padding_idx = 0)

		self.src_embedding.apply(init_weights)
		self.target_embedding.apply(init_weights)

		self.encoder_lstm = nn.LSTM(self.embedding_size , self.hidden_size, bidirectional=True)

		self.encoder_lstm.apply(init_weights)

		decoder_lstm_input = self.embedding_size  + self.hidden_size if self.feed_input else self.embedding_size
		self.decoder_lstm = nn.LSTMCell(decoder_lstm_input, self.hidden_size)

		self.decoder_lstm.apply(init_weights)

		self.att_src_linear = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

		self.att_src_linear.apply(init_weights)

		self.att_vec_linear = nn.Linear(self.hidden_size * 2 + self.hidden_size, self.hidden_size, bias=False)

		self.att_vec_linear.apply(init_weights)

		self.readout = nn.Linear(self.hidden_size, self.target_vocab_size, bias=False)
		self.readout.apply(init_weights)

		self.dropout = nn.Dropout(self.dropout_rate)
		self.decoder_cell_init = nn.Linear(hidden_size * 2, hidden_size)
		self.decoder_cell_init.apply(init_weights)

		# self.loss = F.NLLLoss()


	def forward(self, src_sentences, target_sentences, source_sentence_length, target_sentence_length):
		src_encodings, decoder_init_vec = self.encode(src_sentences, source_sentence_length)

		src_sent_masks = self.get_attention_mask(src_encodings, source_sentence_length)
		att_vecs = self.decode(src_encodings, src_sent_masks, decoder_init_vec, target_sentences)

		tgt_words_log_prob = F.log_softmax(self.readout(att_vecs), dim=-1)

		tgt_words_mask = (target_sentences != 0).float()
		
		tgt_words_log_prob = tgt_words_log_prob.permute(1,0,2)
		target_sentences =  target_sentences.unsqueeze(2)
		# print(tgt_words_log_prob.shape)
		# print(target_sentences.shape)
		tgt_gold_words_log_prob = torch.gather(tgt_words_log_prob, index=target_sentences, dim=-1).squeeze(-1) * tgt_words_mask
		# print(tgt_gold_words_log_prob.shape)
		# print(tgt_words_mask.shape)
		# * tgt_words_mask[1:]

		scores = tgt_gold_words_log_prob.sum(dim=0)

		return scores


	def get_attention_mask(self, src_encodings, src_sentence_length):
		src_sent_masks = torch.zeros(src_encodings.size(0), src_encodings.size(1), dtype=torch.float)

		for e_id, src_len in enumerate(src_sentence_length):
			src_sent_masks[e_id, src_len:] = 1


		return src_sent_masks.to(self.device)


	def encode(self, src_sentences, src_sentence_length):

		src_embeddings = self.src_embedding(src_sentences)

		src_embeddings = src_embeddings.view(src_embeddings.shape[1], src_embeddings.shape[0], src_embeddings.shape[2])
		
		packed_src_embedding = pack_padded_sequence(src_embeddings, src_sentence_length)
	
		src_encodings, (last_state, last_cell) = self.encoder_lstm(packed_src_embedding)
		src_encodings, _ = pad_packed_sequence(src_encodings)
		src_encodings = src_encodings.permute(1, 0, 2)
		dec_init_cell = self.decoder_cell_init(torch.cat([last_cell[0], last_cell[1]], dim=1))
		dec_init_state = torch.tanh(dec_init_cell)

		return src_encodings, (dec_init_state, dec_init_cell)

	def decode(self, src_encodings, src_sent_masks, decoder_init_vec, target_sentences):
		
		src_encoding_att_linear = self.att_src_linear(src_encodings)
		batch_size = src_encodings.size(0)
		att_tm1 = torch.zeros(batch_size, self.hidden_size, device=self.device)

		tgt_word_embeds = self.target_embedding(target_sentences)

		h_tm1 = decoder_init_vec

		att_ves = []
		# 
		tgt_word_embeds = tgt_word_embeds.permute(1,0,2)

		for y_tm1_embed in tgt_word_embeds.split(split_size=1):
			y_tm1_embed = y_tm1_embed.squeeze(0)

			if self.feed_input:
				x = torch.cat([y_tm1_embed, att_tm1], dim=-1)
			else:
				x = y_tm1_embed

			(h_t, cell_t), att_t, alpha_t = self.step(x, h_tm1, src_encodings, src_encoding_att_linear, src_sent_masks)
			att_tm1 = att_t
			h_tm1 = h_t, cell_t
			att_ves.append(att_t)
		att_ves = torch.stack(att_ves)
		return att_ves

	def dot_prod_attention(self, h_t: torch.Tensor, src_encoding: torch.Tensor, src_encoding_att_linear: torch.Tensor,
                           mask: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # (batch_size, src_sent_len)
		att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)

		if mask is not None:
			att_weight.data.masked_fill_(mask.byte(), -float('inf'))

		softmaxed_att_weight = F.softmax(att_weight, dim=-1)

		att_view = (att_weight.size(0), 1, att_weight.size(1))
        # (batch_size, hidden_size)
		ctx_vec = torch.bmm(softmaxed_att_weight.view(*att_view), src_encoding).squeeze(1)

		return ctx_vec, softmaxed_att_weight

	def step(self, x: torch.Tensor,
			h_tm1: Tuple[torch.Tensor, torch.Tensor],
			src_encodings: torch.Tensor, src_encoding_att_linear: torch.Tensor, src_sent_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:

		h_t, cell_t = self.decoder_lstm(x, h_tm1)

		ctx_t, alpha_t = self.dot_prod_attention(h_t, src_encodings, src_encoding_att_linear, src_sent_masks)

		att_t = torch.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
		att_t = self.dropout(att_t)

		return (h_t, cell_t), att_t, alpha_t



		


		


		

	



