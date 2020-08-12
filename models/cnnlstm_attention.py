import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101


class EncoderAttnCNN(nn.Module):
    def __init__(self):
        super(EncoderAttnCNN, self).__init__()
        self.resnet = resnet101(pretrained=True)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
        self.attn = nn.Linear(300, 300)
        
    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :]) 
                x = x.view(x.size(0), -1) 
            attn_weights = torch.sigmoid(self.attn(x))  
            attn_applied = attn_weights * x      
            cnn_embed_seq.append(attn_applied)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, num_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=2):
        super(DecoderRNN, self).__init__()
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=300,
            hidden_size=256,
            num_layers=3,
        )

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, self.num_classes)

    def forward(self, x):
        self.LSTM.flatten_parameters()
        out, (h_n, h_c) = self.LSTM(x)
        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x
