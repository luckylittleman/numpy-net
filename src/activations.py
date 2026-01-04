import numpy as np

class Activation_ReLU:
 def forward(self,inputs):
     #calculate output from input
    self.inputs=inputs
    self.output=np.maximum(0,inputs)