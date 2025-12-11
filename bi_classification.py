import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# define a simple feedforward neural network for binary classification
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # define layers
        self.fc1 = nn.Linear(2, 4)  # input layer to hidden layer 2-> 4
        self.fc2 = nn.Linear(4, 1)  # hidden layer to output layer 4 -> 1
        self.sigmoid = nn.Sigmoid()  # sigmoid activation for output layer

    # forward pass
    def forward(self, x):
        x = torch.relu(self.fc1(x))    # hidden layer uses ReLU activation
        x = self.sigmoid(self.fc2(x))  # output layer uses Sigmoid activation
        return x

# generate some stochastic binary classification data
n_samples = 100
data = torch.randn(n_samples, 2)    # 100 samples, 2 features

# define labels based on a linear boundary
# .float to convert boolean to float tensor
# unsqueeze(1) to make it a column vector, add one dimension at dim 1(last dimension)
labels = (data[:, 0]**2 + data[:, 1]**2 < 1).float().unsqueeze(1)  # inside circle -> class 1, outside -> class 0


# visualize the data
plt.scatter(data[:, 0], data[:, 1], c=labels.squeeze(), cmap='coolwarm')
plt.title('Binary Classification Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


# define model
model = SimpleNN()
# define loss function and optimizer
criterion = nn.BCELoss()  # binary cross-entropy loss
optimizer = optim.SGD(model.parameters(), lr=0.1)  # stochastic gradient descent

for epoch in range(1000):
    # forward pass
    output = model(data)
    loss = criterion(output, labels)  # compute loss

    # backward pass
    optimizer.zero_grad()  # zero the gradients before backward pass
    loss.backward()        # backward pass: compute gradient
    optimizer.step()       # update weights
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{100}], Loss: {loss.item():.4f}')

# ===================use model to make predictions===================

plt.scatter(data[:, 0], data[:, 1], c=(output.detach().numpy() > 0.5).squeeze(), cmap='coolwarm')
plt.title("Decision Boundary")
plt.show()
