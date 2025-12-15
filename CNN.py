# import library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============== load data
# transform
transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5,), (0.5,))   # mean 0.5(减去0.5), std 0.5([-0.5, 0.5]缩放到[-1,1]])
])
# x' = (x - mean)/std   原来是[0, 1]变成[-1, 1]

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=64, shuffle = False)

# =============== define CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # define convolutional layers 1 (1 input , 32 output channels, 3x3 kernel)
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, stride = 1, padding = 1)
        # define convolutional layers 2 (32 input , 64 output channels, 3x3 kernel)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)
        # define a fully connected layer
        self.fc1 = nn.Linear(64 * 7 * 7, 128)       # after 2 maxpooling, image size is reduced to 7x7
        self.fc2 = nn.Linear(128, 10) # 10 output classes for MNIST

    def forward(self, x):
        x = F.relu(self.conv1(x))  # apply conv1 + relu
        x = F.max_pool2d(x, 2)     # apply 2x2 max pooling

        x = F.relu(self.conv2(x))  # apply conv2 + relu
        x = F.max_pool2d(x, 2)     # apply 2x2 max pooling

        x = x.view(-1, 64 * 7 * 7) # flatten the tensor

        x = F.relu(self.fc1(x))       # fully connected layer + relu
        x = self.fc2(x)               # output layer
        return x

# create model instance
model= SimpleCNN()

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)

# train the model
num_epochs = 5
model.train()                               # set the model to training mode
for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in train_loader:
        outputs = model(images)             # forward pass
        loss = criterion(outputs, labels)   # compute loss

        optimizer.zero_grad()               # zero the gradients
        loss.backward()                     # backward pass
        optimizer.step()                    # update weights

        total_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

# evaluate the model
model.eval()
correct = 0
total = 0

with torch.no_grad():                                       # close gradient calculation
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()       # count correct predictions

accuracy = 100 * correct / total
print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

# visualize some predictions
dataiter = iter(test_loader)
# get a batch of test images
images, labels = next(dataiter)
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# create 1 row 6 columns subplots, size 12x4
fig, axes = plt.subplots(1, 6, figsize = (12, 4))
for i in range(6):
    axes[i].imshow(images[i][0], cmap='gray')               # only one channel, show image in subplot
    axes[i].set_title(f"Label: {labels[i]}\nPred: {predicted[i]}")  # set title
    axes[i].axis('off')                                     # remove axis
plt.show()
