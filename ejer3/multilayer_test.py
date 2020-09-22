import numpy as np
import pickle
import random
import math
from MultilayerPerceptron import MultilayerPerceptron

#input
_input = [[0,0], [0,1], [1, 0], [1, 1]]
_expected = [[0], [0], [0], [1]]

learning_rate = 0.01
momentum = 0.9
test_p = 0.25

beta = 0.7 #TODO: ver que valor poner

#activación no lineal y su derivada
def logistic(x):
    return 1 / (1 + np.exp(-2 * beta * x))

def logistic_d(x):
    act = logistic(x)
    return 2 * beta * act * (1 - act)

nn = MultilayerPerceptron(learning_rate, momentum, logistic, logistic_d, test_p)

nn.entry_layer(2)
nn.add_hidden_layer(2)
nn.output_layer(1)

error = 1
while error > 0.0001:
    error = nn.train(_input, _expected)
    print(error)

print(nn.predict(_input[1]))
