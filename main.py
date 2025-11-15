import numpy as np

# ============================================================================
# Step 1: Create Dataset (XOR Problem)
# ============================================================================
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

print("=== STEP 1: DATASET ===")
print("Input shape:", X.shape)  # (4, 2)
print("Output shape:", y.shape) # (4, 1)
print("\nDataset:")
for i in range(len(X)):
    print(f"Input: {X[i]} -> Output: {y[i]}")
print()

# ============================================================================
# Step 2: Initialize Network Architecture
# ============================================================================
np.random.seed(42)  # For reproducible results

input_size = 2
hidden_size = 4
output_size = 1

# Initialize weights with LARGER random values for better learning
W1 = np.random.randn(input_size, hidden_size) * 0.5  # Increased from 0.1
W2 = np.random.randn(hidden_size, output_size) * 0.5 # Increased from 0.1

# Initialize biases to zeros
b1 = np.zeros((1, hidden_size))  # Shape: (1, 4)
b2 = np.zeros((1, output_size))  # Shape: (1, 1)

print("=== STEP 2: NETWORK ARCHITECTURE ===")
print(f"Network: {input_size} inputs → {hidden_size} hidden → {output_size} output")
print("W1 shape:", W1.shape)
print("b1 shape:", b1.shape)
print("W2 shape:", W2.shape)
print("b2 shape:", b2.shape)
print("\nInitial weights W1 (input → hidden):")
print(W1)
print()

# ============================================================================
# Step 3: Forward Pass & Activation Functions
# ============================================================================
def sigmoid(x):
    """The sigmoid activation function - squishes values between 0 and 1"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid - needed for backpropagation"""
    return x * (1 - x)

def forward_pass(X, W1, b1, W2, b2):
    """
    Forward pass through the network
    Returns: Z1, A1, Z2, A2 (we need these for backpropagation)
    """
    # Input → Hidden
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    
    # Hidden → Output
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    
    return Z1, A1, Z2, A2

print("=== STEP 3: FORWARD PASS (INITIAL) ===")
Z1, A1, Z2, A2 = forward_pass(X, W1, b1, W2, b2)

print("Hidden layer activation A1 shape:", A1.shape)
print("Output predictions A2 shape:", A2.shape)
print("\nBefore training:")
for i in range(len(X)):
    print(f"Input: {X[i]} -> Prediction: {A2[i][0]:.4f} (true: {y[i][0]})")
print()

# ============================================================================
# Step 4: Backpropagation & Loss Functions
# ============================================================================
def compute_loss(y_true, y_pred):
    """Calculate Mean Squared Error loss"""
    return np.mean(np.square(y_true - y_pred))

def backward_pass(X, y, Z1, A1, Z2, A2, W2):
    """
    Compute gradients of loss with respect to weights and biases
    Returns: dW1, db1, dW2, db2
    """
    m = X.shape[0]  # Number of samples
    
    # Output layer gradients
    dA2 = (A2 - y)
    dZ2 = dA2 * sigmoid_derivative(A2)
    
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    # Hidden layer gradients
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    return dW1, db1, dW2, db2

print("=== STEP 4: BACKPROPAGATION ===")
initial_loss = compute_loss(y, A2)
print(f"Initial Loss: {initial_loss:.4f}")

dW1, db1, dW2, db2 = backward_pass(X, y, Z1, A1, Z2, A2, W2)

print("Gradient shapes (same as weights):")
print("dW1 shape:", dW1.shape)
print("db1 shape:", db1.shape)
print("dW2 shape:", dW2.shape)
print("db2 shape:", db2.shape)
print("\nSample gradient dW1 (should be larger now):")
print(dW1)
print(f"Average gradient magnitude: {np.mean(np.abs(dW1)):.6f}")
print()

# ============================================================================
# Step 5: Training Loop
# ============================================================================
print("=== STEP 5: TRAINING LOOP ===")

epochs = 20000           # More epochs
learning_rate = 1.0      # Much larger learning rate
loss_history = []

for epoch in range(epochs):
    # Forward pass
    Z1, A1, Z2, A2 = forward_pass(X, W1, b1, W2, b2)
    
    # Compute loss
    loss = compute_loss(y, A2)
    loss_history.append(loss)
    
    # Backward pass
    dW1, db1, dW2, db2 = backward_pass(X, y, Z1, A1, Z2, A2, W2)
    
    # Update weights and biases
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    # Print progress every 2000 epochs
    if epoch % 2000 == 0:
        grad_magnitude = np.mean(np.abs(dW1))
        print(f"Epoch {epoch:5d} | Loss: {loss:.8f} | Avg|dW1|: {grad_magnitude:.8f}")

# Final results
print("\n=== TRAINING COMPLETE ===")
print(f"Final Loss: {loss_history[-1]:.8f}")
print(f"Loss decreased: {loss_history[0]:.4f} → {loss_history[-1]:.8f}")

# Show final predictions
Z1, A1, Z2, A2 = forward_pass(X, W1, b1, W2, b2)
print("\nFinal Predictions:")
print("-" * 50)
for i in range(len(X)):
    pred = A2[i][0]
    target = y[i][0]
    binary = 1 if pred > 0.5 else 0
    correct = "✓" if binary == target else "✗"
    print(f"Input: {X[i]} | Pred: {pred:.4f} | Target: {target} | {correct}")

# Calculate accuracy
predictions_binary = (A2 > 0.5).astype(int)
accuracy = np.mean(predictions_binary == y) * 100
print(f"\nFinal Accuracy: {accuracy:.1f}%")