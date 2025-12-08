import torch
import torch.nn as nn
import torch.optim as optim

# define a simpole fully connected nn
class SimpleNN(nn.Module):         #extend nn.Module
		def __init__(self):
				super(SimpleNN, self).__init__()
				self.fc1 = nn.Linear(2, 2)    # input to hidden layer   2 input, 2 output
				self.fc2 = nn.Linear(2, 1)    # hidden layer to output  2 input, 1 output

		def forward(self, x):
				# relu let the negative values become zero
				x = torch.relu(self.fc1(x))     #use fc1 to process input, then apply relu activation
				x = self.fc2(x)                 #use fc2 to process hidden layer to output
				return x

# create nn sample
model = SimpleNN()
print(model)

# ===================================== forward propagation sample =====================================
# rand input
x = torch.randn(1, 2)

# forward propagation
output = model(x)
print(output)

# define the error, now use the MSE(mean square error)
criterion = nn.MSELoss()

# set the target, assume is 1
target = torch.randn(1, 1)

# compute loss
loss = criterion(output, target)
print(loss)

# ===================================== backward propagation sample =====================================
optimizer = optim.Adam(model.parameters(), lr=0.001)

#step
for epoch in range(100):
    optimizer.zero_grad()    # everytime iteration, clean the gradient
    output = model(x)
    loss = criterion(output, target)
    loss.backward()          # backward, calculate the grad
    optimizer.step()         # update the parameter of the nn model

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
        break

for param in model.parameters():
    print(param)
