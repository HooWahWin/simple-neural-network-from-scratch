import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    def forward(self, input):
        pass
    def backward(self, output_error, learning_rate):
        pass

# Dense Layer
class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    # forward propagation
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, input) + self.bias
    

    # backward propagation
    def backward(self, output_error, learning_rate):
        weight_error = np.dot(output_error, self.input.T)
        input_error = np.dot(self.weights.T, output_error)

        self.weights -= learning_rate * weight_error
        self.bias -= learning_rate * output_error

        return input_error
    
# Flatten Layer
class Flatten(Layer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
    
    # forward propagation
    def forward(self, input):
        self.input = input
        return np.reshape(input, self.output_size)
    
    # backward propagation
    def backward(self, output_error, learning_rate):
        return np.reshape(output_error, self.input_size)