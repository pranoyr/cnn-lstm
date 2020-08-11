import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from  torchvision import transforms
import argparse
import tensorboardX
import os
import random
import numpy as np
from utils import AverageMeter, calculate_accuracy


def val_epoch(encoder_cnn, decoder_rnn, data_loader, criterion, opt):
    encoder_cnn.eval()
    decoder_rnn.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()
    with torch.no_grad():
        for (data, targets) in data_loader:
            data, targets = data.to(opt.device), targets.to(opt.device)
            out_cnn = encoder_cnn(data)
            outputs = decoder_rnn(out_cnn)  

            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), data.size(0))
            accuracies.update(acc, data.size(0))

    # show info
    print('Validation set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(len(data_loader.dataset), losses.avg, accuracies.avg * 100))
    return losses.avg, accuracies.avg

    