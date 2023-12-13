import numpy as np
from layer import Layer

class Activation(Layer):
    def __init__ (self, function, function_prime):
        self.function = function
        self.function_prime = function_prime

    def forward(self, input):
        self.input = input
        return self.function(input)
    
    def backward(self, output_error, learning_rate):
        return np.multiply(self.function_prime(self.input), output_error)
    
# Sigmoid Activation
class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)
        
        super().__init__(sigmoid, sigmoid_prime)

# Softmax Activation
class Softmax(Layer):

    # forward propagation
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    # backward propagation
    def backward(self, output_error, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_error)