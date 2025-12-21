
import numpy as np
from src.layers import Layer_Dense

X = np.array([[1.0, 2.0, 3.0],
              [2.0, 5.0, -1.0],
              [-1.5, 2.7, 3.3]])

dense1 = Layer_Dense(n_inputs=3, n_neurons=4)
dense1.forward(X)

print("Output:\n", dense1.output)