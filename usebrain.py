import numpy as np
import pickle
from src.layers import Layer_Dense
from src.activations import Activation_ReLU, Activation_Softmax

# 1. Setup the "Body" (The Network Structure)
# We must build the same structure (3 inputs -> 3 hidden -> 3 outputs)
dense1 = Layer_Dense(n_inputs=3, n_neurons=3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(n_inputs=3, n_neurons=3)
activation2 = Activation_Softmax()

# 2. Load the "Brain" (The Weights)
print("Loading brain from disk...")
with open('brain.pkl', 'rb') as f:
    saved_data = pickle.load(f)

# Inject the saved memories into the new layers
dense1.weights = saved_data['dense1_weights']
dense1.biases = saved_data['dense1_biases']
dense2.weights = saved_data['dense2_weights']
dense2.biases = saved_data['dense2_biases']

print("Brain loaded and ready!")

# 3. Make a Prediction on NEW Data
# Let's pretend this is a new student:
# [Hours Studied=2.0, Hours Slept=5.0, Attendance=-1.0] -> This looks like Class 1 (Pass) in our data
new_data = np.array([[2.0, 5.0, -1.0]])

# Forward Pass (No training! Just thinking.)
dense1.forward(new_data)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print("\nPrediction for new data:")
print(activation2.output)
print("Predicted Class:", np.argmax(activation2.output))