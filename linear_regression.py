import torch
import numpy as np
import matplotlib.pyplot as plt

# =============================== Generate synthetic data ===============================
# random seed for the same for each running
torch.manual_seed(42)

# generate training data
X = torch.randn(100, 2)

# set the true weight and bias
true_w = torch.tensor([2.0, 3.0])
true_b = 4.0

# add noise at the data, Y is the real value with noise
Y = X @ true_w + true_b + torch.randn(100)*0.1

print(X[:5])
print(Y[:5])

# =============================== define the model ===============================
import torch.nn as nn
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear_fc = nn.Linear(2, 1) # input_dim=2, output_dim=1

    def forward(self, x):
        return self.linear_fc(x)

# creaate the model instance
model = LinearRegressionModel()

# =============================== define loss function and optimizer ============
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# =============================== training the model ===============================
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()    # set the model to training mode

    # forward pass
    outputs = model(X)
    loss = criterion(outputs.squeeze(), Y)

    # backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}')

# =============================== evaluate the model ===============================
print("Trained weights:", model.linear_fc.weight.data)
print("Trained bias:", model.linear_fc.bias.data)

# make predictions on the new data
with torch.no_grad():
    predictions = model(X)

plt.scatter(X[:, 0], Y, color='red', label='True values')
plt.scatter(X[:, 0], predictions, color='blue', label='Predictions')
plt.legend()
plt.show()
