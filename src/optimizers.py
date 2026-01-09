class Optimizer_SGD:
    #initialize a learning rate of 1.0
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self,layers):
       #update the weights and biases for a given layer using the gradients calculated during backprop.
       #weights=weights - learning_rate * dweights
       layers.weights +=-self.learning_rate * layers.dweights

       #bias=bias - learning_rate * dbias
       layers.biases +=-self.learning_rate * layers.dbiases