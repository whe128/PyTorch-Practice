import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MyDataset(Dataset):
		def __init__(self, X_data, Y_data):
				# is list or array
				# x input feature
				# y output feature
				self.X_data = X_data
				self.Y_data = Y_data

		def __len__(self):
				# return the length of dataset
				return len(self.X_data)

		def __getitem__(self, idx):
				# return the data from the idx
				# transfer to tensor
				x = torch.tensor(self.X_data[idx], dtype = torch.float32)
				y = torch.tensor(self.Y_data[idx], dtype = torch.float32)
				return x, y

# example data
X_data = [[1, 2], [3, 4], [5, 6], [7, 8]]
Y_data = [1, 0, 1, 0]

# create dataset instance
dataset = MyDataset(X_data, Y_data)
# everytime, load batch size
# shuffle: whether shuffle the date order
dataloader = DataLoader(dataset, batch_size = 2, shuffle = False)

for epoch in range(3):
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            print(f'Batch {batch_idx + 1}:')
            print(f'Inputs: {inputs}')
            print(f'Labels: {labels}')
