import torch
from torch import nn

from models import cnnlstm, cnnlstm_attention

def generate_model(opt):
	assert opt.model in [
		'cnnlstm', 'cnnlstm_attn'
	]

	if opt.model == 'cnnlstm':
		encoder, decoder = cnnlstm.EncoderCNN(), cnnlstm.DecoderRNN()
	elif opt.model == 'cnnlstm_attn':
		encoder, decoder = cnnlstm_attention.EncoderAttnCNN(), cnnlstm_attention.DecoderRNN()
	return encoder, decoder