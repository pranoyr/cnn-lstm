import torch
from torch.utils import data
from PIL import Image
import os

class Dataset(data.Dataset):
	'Characterizes a dataset for PyTorch'
	def __init__(self, list_IDs, labels, transform=None):
		'Initialization'
		self.labels = labels
		self.list_IDs = list_IDs
		self.transform = transform

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.list_IDs)

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
		X = X[:100]

		X = torch.stack(X)
		y = self.labels[path]
		return X, y