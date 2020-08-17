import torch
from torch import nn

from models import cnnlstm

def generate_model(opt, device):
	assert opt.model in [
		'cnnlstm'
	]

	if opt.model == 'cnnlstm':
		model = cnnlstm.CNNLSTM()
	return model.to(device)