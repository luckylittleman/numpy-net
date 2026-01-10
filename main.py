import numpy as np
from src.datasets import create_data_spiral # <--- New Import
from src.layers import Layer_Dense
from src.activations import Activation_ReLU
from src.loss import Activation_Softmax_Loss_CategoricalCrossentropy
from src.optimizers import Optimizer_SGD


# 1. Create Data
X, y = create_data_spiral(samples=100, classes=3)

# 2. Initialize Network - NOW WITH 2 HIDDEN LAYERS
# Layer 1: Input (2) -> Hidden (64)
dense1 = Layer_Dense(n_inputs=2, n_neurons=64) 
activation1 = Activation_ReLU()

# Layer 2: Hidden (64) -> Hidden (64)  <--- NEW INTERMEDIATE LAYER
dense2 = Layer_Dense(n_inputs=64, n_neurons=64) 
activation2 = Activation_ReLU()

# Layer 3: Hidden (64) -> Output (3)   <--- OUTPUT LAYER
dense3 = Layer_Dense(n_inputs=64, n_neurons=3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# 3. Optimizer (Lower the rate slightly to 0.5 to be safe)
optimizer = Optimizer_SGD(learning_rate=1.0) # Keep 1.0 or try 0.5 if unstable

# 4. Training Loop
print("Starting Deep Training...")
for epoch in range(10001):
    
    # --- Forward Pass ---
    dense1.forward(X)
    activation1.forward(dense1.output)
    
    # Pass through the NEW layer
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    # Output layer
    dense3.forward(activation2.output)
    
    loss = loss_activation.forward(dense3.output, y)
    
    # Calculate Accuracy
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if epoch % 1000 == 0:
        print(f'Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

    # --- Backward Pass ---
    loss_activation.backward(loss_activation.output, y)
    dense3.backward(loss_activation.dinputs)      # Back from Output
    activation2.backward(dense3.dinputs)          # Back through ReLU
    dense2.backward(activation2.dinputs)          # Back through Hidden 2 (NEW)
    activation1.backward(dense2.dinputs)          # Back through ReLU
    dense1.backward(activation1.dinputs)          # Back through Hidden 1

    # --- Optimization ---
    optimizer.update_params(dense1)
    optimizer.update_params(dense2) # Update the new layer too!
    optimizer.update_params(dense3)
    # ... (Previous training loop code) ...
# ... (Previous training loop code) ...

# --- VISUALIZATION ---
import matplotlib
matplotlib.use('Agg') # <--- 1. This tells Python "Don't try to open a window"
import matplotlib.pyplot as plt

print("Generating visualization...")

# 1. Create a mesh grid
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# 2. Flatten grid
mesh_data = np.c_[xx.ravel(), yy.ravel()]

# 3. Forward Pass on the Grid
dense1.forward(mesh_data)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)

# 4. Get predictions
Z = np.argmax(dense3.output, axis=1)
Z = Z.reshape(xx.shape)

# 5. Plot and SAVE
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors='k')
plt.title(f"Spiral Decision Boundary (Accuracy: {accuracy*100:.2f}%)")

print("Saving image to 'spiral_result.png'...")
plt.savefig('spiral_result.png') # <--- 2. Save instead of Show
print("Done! Check your folder.")