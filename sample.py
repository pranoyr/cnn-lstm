
import torchvision.models as models
from torchvision import models
import torch.nn.functional as F

import torch.nn as nn
import torch

x1 = torch.tensor([[1,2,3],[1,2,3]], dtype=torch.float)
x = torch.tensor([[1,2,3],[1,2,3]], dtype=torch.float)
x = F.sigmoid(x)


input = torch.randn(10, 3, 4)

hidden = torch.Tensor([[11,11,11,11,11],
                      [1,2,3,4,5]])
print(hidden)

res = torch.dot(hidden, hidden)

print(res)