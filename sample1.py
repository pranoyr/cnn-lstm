import torch

torch.Tensor(1,1)

my_list = [torch.Tensor(1,1),torch.Tensor(1,1),torch.Tensor(1,1),torch.Tensor(1,1), torch.Tensor(1,1), torch.Tensor(1,1), torch.Tensor(1,1), torch.Tensor(1,1)]
  
# How many elements each 
# list should have 
n = 2
  
# using list comprehension 
final = [torch.Tensor(my_list[i * n:(i + 1) * n]) for i in range((len(my_list) + n - 1) // n ) if len(my_list[i * n:(i + 1) * n]) == 2]  
print (final) 

print(torch.stack(final).size())
