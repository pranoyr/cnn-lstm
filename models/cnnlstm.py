import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101


class CNNLSTM(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(CNNLSTM, self).__init__()
        self.resnet = resnet101(pretrained=True)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
       
    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  
                x = x.view(x.size(0), -1).unsqueeze(0)
                out, hidden = self.lstm(x, hidden)         

     

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
        out, (h_n, h_c) = self.LSTM(x)
        # FC layers
        # choose RNN_out at the last time step
        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        #x = F.dropout(x, p=0.3)
        x = self.fc2(x)
        return x
