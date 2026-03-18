import torch
import torch.nn as nn
import torch.optim as optim

# Simple CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3)
        self.fc = nn.Linear(26*26, 1)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(-1, 26*26)
        x = torch.sigmoid(self.fc(x))
        return x

model = CNN()

# Dummy image
x = torch.randn(1,1,28,28)
y = torch.tensor([[1.0]])

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Forward pass
output = model(x)
loss = criterion(output, y)

# Backpropagation
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Loss:", loss.item())
