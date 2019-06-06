import torch
 
# x = torch.randn(2)

# y = torch.randn(2)

# y=torch.Tensor([1,2])

# # l = []

# # l.append(x)
# # l.append(y)


# # z = torch.stack(l)

# # print(z)


# print(y)

# import torch.nn as nn
# import torch

input_5d = torch.randn(2,2)
print(input_5d)
y  = input_5d.view(-1,)
print(y)
# print(input)

# input = input[-1,:]

# print(input)


# for t in range(input_5d.size(1)):
#             # ResNet CNN
#             with torch.no_grad():
#                 x = input_5d[:, t, :, :, :]  # ResNet
#                 print(x.size())
#                 x = x.view(x.size(0), -1) # flatten output of conv
#                 print(x.size())