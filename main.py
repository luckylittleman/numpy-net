
import numpy as np
from src.layers import Layer_Dense
from src.activations import Activation_ReLU,Activation_Softmax
from src.loss import Loss_CategoricalCrossentropy


X = np.array([[1.0, 2.0, 3.0],
              [2.0, 5.0, -1.0],
              [-1.5, 2.7, 3.3]])

#define the target(Actual Values)
#lets assume the correct classes are:class 0,Class 1,Class 1
y=np.array([0,1,1])

#creating objects
dense1 = Layer_Dense(n_inputs=3, n_neurons=3)
activation1=Activation_ReLU()

dense2=Layer_Dense(n_inputs=3,n_neurons=3)
activation2=Activation_Softmax()
loss_function=Loss_CategoricalCrossentropy()

#forward pass
dense1.forward(X)#linear step
activation1.forward(dense1.output)#non-linear step(takes the dense output as input)

dense2.forward(activation1.output)#Input to dense2 is output of activation1
activation2.forward(dense2.output)

#calculate Loss
loss=loss_function.calculate(activation2.output,y)

print("Final Probabilities:\n", activation2.output)
print("\nLoss:",loss)
print("\nSum of rows (Should be close to 1):\n",np.sum(activation2.output,axis=1))

