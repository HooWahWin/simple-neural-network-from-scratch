import numpy as np

from layer import Dense, Flatten
from activation import Sigmoid, Softmax, ReLu
from loss import categorical_cross_entropy, categorical_cross_entropy_prime

from keras.datasets import mnist
from keras.utils import to_categorical

# load the MNIST dataset
def preprocess_data(x, y, limit):
    # normalize input data
    x = x.astype("float32") / 255

    # one hot encode the labels
    y = to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)

    return x[:limit], y[:limit]


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 2000)
x_test, y_test = preprocess_data(x_test, y_test, 100)


LEARNING_RATE = 0.25
EPOCHS = 100
LOSS, LOSS_PRIME = categorical_cross_entropy, categorical_cross_entropy_prime

# create the model
model = [
    Flatten((28, 28), (28 * 28, 1)),
    Dense(28 * 28, 100),
    Sigmoid(),
    Dense(100, 10),
    Softmax()
]

for e in range(EPOCHS):
    error = 0

    for x, y in zip(x_train, y_train):
        # forward propagation
        output = x
        for layer in model:
            output = layer.forward(output)

        # error
        error += LOSS(y, output)

        # backward propagation
        output_grad = LOSS_PRIME(y, output)
        for layer in reversed(model):
            output_grad = layer.backward(output_grad, LEARNING_RATE)

    print(f"Epoch: {e + 1}, Error: {error/len(x_train)}")

# test the model
counter = 0
for x, y in zip(x_test, y_test):
    output = x
    for layer in model:
        output = layer.forward(output)
   
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
    if (np.argmax(output) == np.argmax(y)):
        counter += 1

print(f"Accuracy: {(counter / len(x_test)) * 100.00}")