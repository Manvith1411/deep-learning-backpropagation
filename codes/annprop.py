import numpy as np

# Input and target
X = np.array([[1, 2]])
y = np.array([[1]])

# Initialize weights
W1 = np.random.rand(2, 2)
W2 = np.random.rand(2, 1)

lr = 0.1

# Forward pass
h = np.dot(X, W1)
h_act = 1/(1+np.exp(-h))       # sigmoid

output = np.dot(h_act, W2)
y_pred = 1/(1+np.exp(-output))

# Loss
loss = np.mean((y - y_pred)**2)

# Backpropagation
d_output = (y_pred - y) * y_pred * (1 - y_pred)

dW2 = np.dot(h_act.T, d_output)

d_hidden = np.dot(d_output, W2.T) * h_act * (1 - h_act)

dW1 = np.dot(X.T, d_hidden)

# Weight update
W2 -= lr * dW2
W1 -= lr * dW1

print("Prediction:", y_pred)
print("Loss:", loss)
