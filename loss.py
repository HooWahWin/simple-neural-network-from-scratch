import numpy as np

# calculates the catergorical cross entropy loss
def categorical_cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss

# calculates the catergorical cross entropy loss derivative 
def categorical_cross_entropy_prime(actual, predicted):
    return -np.divide(actual, predicted)