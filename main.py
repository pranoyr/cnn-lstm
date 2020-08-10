from train import train_epoch
from torch.utils.data import DataLoader
from validation import val_epoch
from opts import parse_opts
from torch.optim import lr_scheduler
from utils import *
from dataset import UF101Dataset
from model import EncoderCNN, DecoderRNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import tensorboardX
import os
import random
import numpy as np


if __name__ == "__main__":
	opt = parse_opts()
	print(opt)

	seed = 1
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	# CUDA for PyTorch
	use_cuda = torch.cuda.is_available()
	device = torch.device(f"cuda:{opt.gpu}" if opt.use_cuda else "cpu")

	# Parameters
	# params = {'batch_size': 4,
    #       'shuffle': True,
    #       'num_workers': 0}

	# Datasets
	partition, labels = load_data(opt.dataset)

	# preprocesing
	transform = transforms.Compose([transforms.Resize([256, 342]),
									transforms.ToTensor(),
									transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

	# generators
	train_data = UF101Dataset(partition['train'], labels, transform)
	val_data = UF101Dataset(partition['val'], labels, transform)
	train_loader = DataLoader(
		train_data, batch_size=1, shuffle=False, collate_fn=train_data.my_collate)
	val_loader = DataLoader(
		val_data, batch_size=1, shuffle=False, collate_fn=val_data.my_collate)

	# tensorboard
	summary_writer = tensorboardX.SummaryWriter(log_dir='tf_logs')

	# defining model
	encoder_cnn = EncoderCNN().to(device)
	decoder_rnn = DecoderRNN().to(device)

	# optimizer
	crnn_params = list(encoder_cnn.parameters()) + \
		list(decoder_rnn.parameters())
	optimizer = torch.optim.Adam(crnn_params, lr=opt.lr_rate)

	scheduler = lr_scheduler.ReduceLROnPlateau(
		optimizer, 'min', patience=opt.lr_patience)
	criterion = nn.CrossEntropyLoss()

	# resume model, optimizer if already exists
	if opt.resume_path:
		checkpoint = torch.load(opt.resume_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		start_epoch = checkpoint['epoch']
		print("Model Restored from Epoch {}".format(start_epoch))
	else:
		start_epoch = 1

	# start training
	for epoch in range(start_epoch, opt.n_epochs + 1):
		train_loss, train_acc = train_epoch(
			encoder_cnn, decoder_rnn, train_loader, criterion, optimizer, epoch, device, opt.log_interval)
		val_loss, val_acc = val_epoch(
			encoder_cnn, decoder_rnn, val_loader, criterion, device)

		# # saving weights to checkpoint
		# if (epoch) % opt.save_interval == 0:
		# 	scheduler.step(val_loss)
		# 	# write summary
		# 	summary_writer.add_scalar(
		# 		'losses/train_loss', train_loss, global_step=epoch)
		# 	summary_writer.add_scalar(
		# 		'losses/val_loss', val_loss, global_step=epoch)
		# 	summary_writer.add_scalar(
		# 		'acc/train_acc', train_acc * 100, global_step=epoch)
		# 	summary_writer.add_scalar(
		# 		'acc/val_acc', val_acc * 100, global_step=epoch)

		# 	state = {'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
		# 			 'optimizer_state_dict': optimizer.state_dict()}
		# 	torch.save(state, os.path.join('snapshots', f'model{epoch}.pth'))
		# 	print("Epoch {} model saved!\n".format(epoch))
