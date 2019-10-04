
import torchvision.models as models
from torchvision import models


import torch.nn as nn
import torch

resnet = models.resnet152(pretrained=False)
modules = list(resnet.children())[:-1]      # delete the last fc layer.

resnet_1 = nn.Sequential(*modules)

x = torch.Tensor(1,3,300,300)

output = resnet_1(x)

for layer in resnet_1:
    x = layer(x)
    print(x.size())


