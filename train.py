import numpy as np
from dataset import *
import argparse
from utils import *
from torch.utils import data
from torchvision import transforms as T
from model import EncoderCNN
from model import DecoderRNN
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./data', help='path to dataset')
opt = parser.parse_args()


#
# Parameters
params = {'batch_size': 4,
          'shuffle': True,
          'num_workers': 1}
learning_rate = 1e-4
log_interval = 2   # interval for displaying training info
epochs = 100



save_model_path = './snapshots'

# Datasets
partition, labels = load_data(opt.data)

# #pre_processing
# transform = transforms.Compose([
#         transforms.RandomHorizontalFlip(), 
#         transforms.Resize((224,224)),
#         transforms.Normalize((0.485, 0.456, 0.406), 
#                              (0.229, 0.224, 0.225)),
#         transforms.ToTensor()])

# preprocesing
transform = transforms.Compose([transforms.Resize([256, 342]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# transform = []
# transform.append(T.Resize(image_size))
# transform.append(T.RandomHorizontalFlip())
# transform.append(T.ToTensor())
# transform.append(T.Normalize(
#     mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
# transform = T.Compose(transform)

# generators
training_set = Dataset(partition['train'], labels, transform)
training_generator = data.DataLoader(training_set, **params, collate_fn=training_set.my_collate)
validation_set = Dataset(partition['val'], labels, transform)
validation_generator = data.DataLoader(validation_set, **params, collate_fn=validation_set.my_collate)

# defining model
encoder_cnn = EncoderCNN()
decoder_rnn = DecoderRNN()

losses = []
scores = []
def train(model, device, training_generator, optimizer, epoch, log_interval):
    # set model as training mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    train_loss = 0.0
    # counting total trained sample in one epoch
    N_count = 0
    correct = 0
    losses = []

    # Training
    for i, (X, y) in enumerate(training_generator):
        # Transfer to GPU
        X, y = X.to(device), y.to(device)

        N_count += X.size(0)

        out_cnn = encoder_cnn(X)
        #out1 = out1.reshape(3, -1, 1000).to(device)
        out_rnn = decoder_rnn(out_cnn)
        loss = F.cross_entropy(out_rnn, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(out_rnn, 1)[1]  # y_pred != output
        correct = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
   
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # show information
        if (i+1) % log_interval == 0:
            avg_loss = train_loss / log_interval
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, N_count, len(training_generator.dataset), 100. * (i + 1) / len(training_generator), avg_loss))
            train_loss = 0.0

    # show information
    acc = 100. * (correct / N_count)
    average_loss = sum(losses)/len(training_generator)
    print('Train set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(N_count, average_loss, acc))
    return average_loss, acc


def validation(model, device, optimizer, validation_generator):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    N_count = 0
    correct = 0
    losses = []
   
    with torch.no_grad():
        for X, y in validation_generator:
            # distribute data to device
            X, y = X.to(device), y.to(device)

            N_count += X.size(0)

            output = rnn_decoder(cnn_encoder(X))

            loss = F.cross_entropy(output, y)
            losses.append(loss)    

            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability
            correct += y_pred.eq(y.view_as(y_pred)).sum().item()

           
    # show information
    acc = 100. * (correct / N_count)
    average_loss = sum(losses)/len(validation_generator)
    print('Validation set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(N_count, average_loss, acc))
    return average_loss, acc


# optimizer
crnn_params = list(encoder_cnn.parameters()) + list(decoder_rnn.parameters())
optimizer = torch.optim.Adam(crnn_params, lr=learning_rate)

# start training
for epoch in range(epochs):
    # train, test model
    train_loss, train_acc = train([encoder_cnn, decoder_rnn], device, training_generator, optimizer, epoch, log_interval)
    val_loss, val_acc = validation([encoder_cnn, decoder_rnn], device, optimizer, validation_generator)

   
    
# def validation(model, device, optimizer, test_loader):
#     # set model as testing mode
#     cnn_encoder, rnn_decoder = model
#     cnn_encoder.eval()
#     rnn_decoder.eval()

#     test_loss = 0
#     all_y = []
#     all_y_pred = []
#     with torch.no_grad():
#         for X, y in test_loader:
#             # distribute data to device
#             X, y = X.to(device), y.to(device).view(-1, )

#             output = rnn_decoder(cnn_encoder(X))

#             loss = F.cross_entropy(output, y, reduction='sum')
#             test_loss += loss.item()                 # sum up batch loss
#             y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

#             # collect all y and y_pred in all batches
#             all_y.extend(y)
#             all_y_pred.extend(y_pred)

#     test_loss /= len(test_loader.dataset)

#     # compute accuracy
#     all_y = torch.stack(all_y, dim=0)
#     all_y_pred = torch.stack(all_y_pred, dim=0)
#     test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

#     # show information
#     print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

#     # save Pytorch models of best record
#     torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
#     torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
#     torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
#     print("Epoch {} model saved!".format(epoch + 1))

# return test_loss, test_score