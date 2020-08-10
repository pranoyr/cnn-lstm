import torch
from torch.utils import data
from PIL import Image
import os

class UF101Dataset(data.Dataset):
	'Characterizes a dataset for PyTorch'
	def __init__(self, list_IDs, labels, transform=None):
		'Initialization'
		self.labels = labels
		self.list_IDs = list_IDs
		self.transform = transform

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.list_IDs)

	def my_collate(self, batch):
		videos = [] 
		targets = []
		for item in batch:
			videos.append(item[0])
			targets.append(item[1])

		videos = torch.cat(videos)
		targets = torch.cat(targets)
		# flatten
		# targets = targets.view(-1)
		targets = targets.type(torch.LongTensor)
		return videos, targets


	def __getitem__(self, index):
		'Generates one sample of data'
		# Select sample
		path = self.list_IDs[index]
		folder = os.listdir(path)
		X = []
		for img_name in folder:
			img = Image.open(os.path.join(path,img_name))
			img = self.transform(img)
			X.append(img)

		n = 16
		X = [torch.stack(X[i * n:(i + 1) * n]) for i in range((len(X) + n - 1) // n ) if len(X[i * n:(i + 1) * n]) == n]
		X = torch.stack(X)

		# y = self.labels[path]
		Y = [self.labels[path] for _ in range(len(X))]
		Y = torch.Tensor(Y)
		return X, Y