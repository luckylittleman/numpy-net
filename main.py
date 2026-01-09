# main.py
import numpy as np
from src.layers import Layer_Dense
from src.activations import Activation_ReLU
from src.loss import Activation_Softmax_Loss_CategoricalCrossentropy
from src.optimizers import Optimizer_SGD

# 1. Create Data (3 samples)
X = np.array([[1.0, 2.0, 3.0],
              [2.0, 5.0, -1.0],
              [-1.5, 2.7, 3.3]])

# Labels: Class 0 (Dog), Class 1 (Cat), Class 1 (Cat)
y = np.array([0, 1, 1])

# 2. Initialize Network
dense1 = Layer_Dense(n_inputs=3, n_neurons=3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(n_inputs=3, n_neurons=3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Initialize Optimizer
optimizer = Optimizer_SGD(learning_rate=0.85)

#training loop
print("=== Training Step ===")
for epoch in range(10001):
    # Forward Pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    #calculate loss
    loss = loss_activation.forward(dense2.output, y)

    #calculate accuracy
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    # Print accuracy every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}, Accuracy: {accuracy}")

    # Backward Pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

print("=== Final Weights and Biases ===")
print("\nFinal predictions:\n", loss_activation.output)

# --- SAVE THE BRAIN ---
import pickle

print("\nSaving the brain...")

# We only need to save the Weights and Biases
# (The structure of the neurons is defined in code, but the 'knowledge' is in these numbers)
saved_data = {
    "dense1_weights": dense1.weights,
    "dense1_biases": dense1.biases,
    "dense2_weights": dense2.weights,
    "dense2_biases": dense2.biases
}

# Write to a binary file named 'brain.pkl'
with open('brain.pkl', 'wb') as f:
    pickle.dump(saved_data, f)

print("Brain successfully saved to 'brain.pkl'!")