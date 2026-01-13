import numpy as np
from src.datasets import create_data_spiral 
from src.layers import Layer_Dense, Layer_Dropout
from src.activations import Activation_ReLU
from src.loss import Activation_Softmax_Loss_CategoricalCrossentropy
from src.optimizers import Optimizer_SGD
from src.optimizers import Optimizer_Adam


# 1. Create Data
X, y = create_data_spiral(samples=100, classes=3)

X_test, y_test = create_data_spiral(samples=100, classes=3)

# 2. Initialize Network - NOW WITH 2 HIDDEN LAYERS
# Layer 1: Input (2) -> Hidden (64)
dense1 = Layer_Dense(n_inputs=2, n_neurons=64) 
activation1 = Activation_ReLU()
dropout1 = Layer_Dropout(0.1)

# Layer 2: Hidden (64) -> Hidden (64)  <--- NEW INTERMEDIATE LAYER
dense2 = Layer_Dense(n_inputs=64, n_neurons=64) 
activation2 = Activation_ReLU()
dropout2 = Layer_Dropout(0.1)

# Layer 3: Hidden (64) -> Output (3)   <--- OUTPUT LAYER
dense3 = Layer_Dense(n_inputs=64, n_neurons=3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# 3. Optimizer (Lower the rate slightly to 0.5 to be safe)
optimizer = Optimizer_Adam(learning_rate=0.01, decay=1e-5)

# 4. Training Loop
print("Starting Deep Training...")
for epoch in range(10001):
    
    # --- Forward Pass ---
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    
    # Pass through the NEW layer
    dense2.forward(dropout1.output)
    activation2.forward(dense2.output)
    dropout2.forward(activation2.output)
    
    # Output layer
    dense3.forward(dropout2.output)
    
    loss = loss_activation.forward(dense3.output, y)
    
    # Calculate Accuracy
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if epoch % 1000 == 0:
        print(f'epoch: {epoch} Loss: {loss:.4f}, Accuracy: {accuracy:.4f}' )

        #validation on the test results
        # --- Validation Forward Pass ---
        dense1.forward(X_test)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        dense3.forward(activation2.output)
        loss_test = loss_activation.forward(dense3.output, y_test)
        predictions_test = np.argmax(loss_activation.output, axis=1)
        if len(y_test.shape) == 2:
            y_test = np.argmax(y_test, axis=1)
        else:
                y_test_inds = y_test
        accuracy_test = np.mean(predictions_test == y_test)
        print(f'--- Validation --- Loss: {loss_test:.4f}, Accuracy: {accuracy_test:.4f}' )
   # --- Backward Pass ---
    loss_activation.backward(loss_activation.output, y)
    dense3.backward(loss_activation.dinputs)
    
    dropout2.backward(dense3.dinputs)       # <--- Back through Dropout
    activation2.backward(dropout2.dinputs)  # <--- Input from Dropout gradient
    dense2.backward(activation2.dinputs)
    
    dropout1.backward(dense2.dinputs)       # <--- Back through Dropout
    activation1.backward(dropout1.dinputs)  # <--- Input from Dropout gradient
    dense1.backward(activation1.dinputs)

    # --- Optimization ---
    optimizer.pre_update_params()#tells adam we are starting a new update

    optimizer.update_params(dense1)
    optimizer.update_params(dense2) # Update the new layer too!
    optimizer.update_params(dense3)

    optimizer.pre_update_params() #this part tells adam we are done with this part
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