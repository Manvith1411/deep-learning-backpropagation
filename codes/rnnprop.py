import torch
import torch.nn as nn
import torch.optim as optim

# Simple RNN
class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=3, hidden_size=5, batch_first=True)
        self.fc = nn.Linear(5,1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:,-1,:])
        return torch.sigmoid(out)

model = RNNModel()

# Dummy sequence (3 timesteps)
x = torch.randn(1,3,3)
y = torch.tensor([[1.0]])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Forward pass
output = model(x)
loss = criterion(output, y)

# Backpropagation Through Time
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Loss:", loss.item())
