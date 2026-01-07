# main.py
import numpy as np
from src.layers import Layer_Dense
from src.activations import Activation_ReLU
from src.loss import Activation_Softmax_Loss_CategoricalCrossentropy

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

# 3. Forward Pass
print("--- Forward Pass ---")
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)

loss = loss_activation.forward(dense2.output, y)
print(f"Loss: {loss}")
print(f"Prediction:\n{loss_activation.output[:3]}") # Print first few predictions

# 4. Backward Pass
print("\n--- Backward Pass ---")
# Backpropagate through Loss + Softmax
loss_activation.backward(loss_activation.output, y)

# Backpropagate through Layer 2
dense2.backward(loss_activation.dinputs)

# Backpropagate through Activation 1
activation1.backward(dense2.dinputs)

# Backpropagate through Layer 1
dense1.backward(activation1.dinputs)

# 5. Check Results
print("Gradients for Dense1 Weights:\n", dense1.dweights)
print("Gradients for Dense1 Biases:\n", dense1.dbiases)