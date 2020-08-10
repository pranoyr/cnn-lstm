from train import train_epoch
from torch.utils.data import DataLoader
from validation import val_epoch
from opts import parse_opts
from torch.optim import lr_scheduler
from dataset import get_training_set, get_validation_set
from model import EncoderCNN, DecoderRNN
from mean import get_mean, get_std
from spatial_transforms import (
	Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
	MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose

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

	# train loader
	opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
	if opt.no_mean_norm and not opt.std_norm:
    		norm_method = Normalize([0, 0, 0], [1, 1, 1])
	elif not opt.std_norm:
		norm_method = Normalize(opt.mean, [1, 1, 1])
	else:
		norm_method = Normalize(opt.mean, opt.std)
	spatial_transform = Compose([
			# crop_method,
			Scale((opt.sample_size, opt.sample_size)),
			# RandomHorizontalFlip(),
			ToTensor(opt.norm_value), norm_method
		])
	temporal_transform = TemporalRandomCrop(16)
	target_transform = ClassLabel()
	training_data = get_training_set(opt, spatial_transform,
										 temporal_transform, target_transform)
	train_loader = torch.utils.data.DataLoader(
			training_data,
			batch_size=opt.batch_size,
			shuffle=True,
			num_workers=opt.num_workers,
			pin_memory=True)

	# validation loader
	spatial_transform = Compose([
			Scale((opt.sample_size,opt.sample_size)),
			#CenterCrop(opt.sample_size),
			ToTensor(opt.norm_value), norm_method
		])
	target_transform = ClassLabel()
	temporal_transform = LoopPadding(16)							 
	validation_data = get_validation_set(
			opt, spatial_transform, temporal_transform, target_transform)
	val_loader = torch.utils.data.DataLoader(
			validation_data,
			batch_size=opt.batch_size,
			shuffle=False,
			num_workers=opt.num_workers,
			pin_memory=True)
	

	# tensorboard
	summary_writer = tensorboardX.SummaryWriter(log_dir='tf_logs')

	# defining model
	encoder_cnn = EncoderCNN().to(device)
	decoder_rnn = DecoderRNN().to(device)

	# optimizer
	crnn_params = list(encoder_cnn.parameters()) + \
		list(decoder_rnn.parameters())
	optimizer = torch.optim.Adam(encoder_cnn.parameters(), lr=0.1)

	scheduler = lr_scheduler.ReduceLROnPlateau(
		optimizer, 'min', patience=opt.lr_patience)
	criterion = nn.CrossEntropyLoss()

	# resume model, optimizer if already exists
	if opt.resume_path:
		checkpoint = torch.load(opt.resume_path)
		encoder_cnn.load_state_dict(checkpoint['encoder_state_dict'])
		decoder_rnn.load_state_dict(checkpoint['decoder_state_dict'])
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

		# saving weights to checkpoint
		if (epoch) % opt.save_interval == 0:
			# scheduler.step(val_loss)
			# write summary
			summary_writer.add_scalar(
				'losses/train_loss', train_loss, global_step=epoch)
			summary_writer.add_scalar(
				'losses/val_loss', val_loss, global_step=epoch)
			summary_writer.add_scalar(
				'acc/train_acc', train_acc * 100, global_step=epoch)
			summary_writer.add_scalar(
				'acc/val_acc', val_acc * 100, global_step=epoch)

			state = {'epoch': epoch + 1, 'encoder_state_dict': encoder_cnn.state_dict(),
					 'decoder_state_dict': decoder_rnn.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
			torch.save(state, os.path.join('snapshots', f'model{epoch}.pth'))
			print("Epoch {} model saved!\n".format(epoch))
