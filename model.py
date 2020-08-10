import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()


        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)

        self.fc1 = nn.Linear(resnet.fc.in_features, 512)
        self.bn1 = nn.BatchNorm1d(512, momentum=0.01)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512, momentum=0.01)
        self.fc3 = nn.Linear(512, 300)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            # with torch.no_grad():
            x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
            x = x.view(x.size(0), -1)             # flatten output of conv

            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=0.2)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
        # cnn_embed_seq: shape=(time_step, batch, input_size)

        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, num_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=2):
        super(DecoderRNN, self).__init__()
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=300,
            hidden_size=256,
            num_layers=3,
            # input & output will has batch size as 1s dimension. e.g. (time_step, batch, input_size)
        )

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, self.num_classes)

    def forward(self, x):
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size)
            None represents zero initial hidden state. RNN_out has shape=(time_step, batch, output_size)
        """
        self.LSTM.flatten_parameters()
        out, (h_n, h_c) = self.LSTM(x, None)
        # FC layers
        # choose RNN_out at the last time step
        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        #x = F.dropout(x, p=0.3)
        x = self.fc2(x)
        return x
