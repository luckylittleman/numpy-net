
import numpy as np
from src.layers import Layer_Dense
from src.activations import Activation_ReLU


X = np.array([[1.0, 2.0, 3.0],
              [2.0, 5.0, -1.0],
              [-1.5, 2.7, 3.3]])

#creating objects
dense1 = Layer_Dense(n_inputs=3, n_neurons=4)
activation1=Activation_ReLU()

#forward pass
dense1.forward(X)#linear step
activation1.forward(dense1.output)#non-linear step(takes the dense output as input)


print("Dense Output((Linear):\n", dense1.output)
print("\nReLU Output((non-Linear):\n", activation1.output)
