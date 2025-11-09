import numpy as np

# Create our XOR dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

print("Input shape:", X.shape)  # (4, 2) - 4 samples, 2 features each
print("Output shape:", y.shape) # (4, 1) - 4 samples, 1 output each
print("\nDataset:")
for i in range(len(X)):
    print(f"Input: {X[i]} -> Output: {y[i]}")


# Set a random seed for reproducibility
np.random.seed(42)

# Network architecture
input_size = 2
hidden_size = 4
output_size = 1

# Initialize weights randomly (small values work better)
# Weights between input and hidden layer
W1 = np.random.randn(input_size, hidden_size) * 0.1
# Weights between hidden and output layer
W2 = np.random.randn(hidden_size, output_size) * 0.1

# Initialize biases to zeros
b1 = np.zeros((1, hidden_size))  # bias for hidden layer
b2 = np.zeros((1, output_size))  # bias for output layer

print("W1 shape:", W1.shape)  # (2, 4)
print("b1 shape:", b1.shape)  # (1, 4)
print("W2 shape:", W2.shape)  # (4, 1)
print("b2 shape:", b2.shape)  # (1, 1)

print("\nInitial weights W1 (input → hidden):")
print(W1)