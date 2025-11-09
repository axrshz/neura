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