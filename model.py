import torch
from torch import nn

from models import cnnlstm, cnnlstm_attention

def generate_model(opt, device):
	assert opt.model in [
		'cnnlstm'
	]

	if opt.model == 'cnnlstm':
		encoder, decoder = cnnlstm.EncoderCNN(), cnnlstm.DecoderRNN(num_classes=opt.n_classes)
	return encoder.to(device), decoder.to(device)