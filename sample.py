
import torchvision.models as models
from torchvision import models
import torch.nn.functional as F

import torch.nn as nn
import torch

# resnet = models.resnet152(pretrained=False)
# modules = list(resnet.children())[:-1]      # delete the last fc layer.

# resnet_1 = nn.Sequential(*modules)

# x = torch.Tensor(1,3,300,300)

# output = resnet_1(x)

# for layer in resnet_1:
#     x = layer(x)
#     print(x.size())

x1 = torch.tensor([[1,2,3],[1,2,3]], dtype=torch.float)
x = torch.tensor([[1,2,3],[1,2,3]], dtype=torch.float)
x = F.sigmoid(x)

# print(x.shape)
# print(x1.shape)

# print(x1.unsqueeze(0).shape)
# print(x.unsqueeze(0).shape)

# attn_applied = torch.bmm(x1.unsqueeze(0), x.unsqueeze(0))  
# print(x)
# print(x.shape)


input = torch.randn(10, 3, 4)

# attn_weights = torch.Tensor([[1,0,0,0,0],
#                 [0,1,0,0,0]])

# print(attn_weights)
hidden = torch.Tensor([[11,11,11,11,11],
                      [1,2,3,4,5]])
print(hidden)

res = torch.dot(hidden, hidden)

print(res)