import torch
from torch import nn

from models import cnnlstm

def generate_model(opt, device):
	assert opt.model in [
		'cnnlstm'
	]

	if opt.model == 'cnnlstm':
		model = cnnlstm.CNNLSTM(num_classes=opt.n_classes)
	return model.to(device)