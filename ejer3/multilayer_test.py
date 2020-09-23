import numpy as np
import pickle
import random
import math
from MultilayerPerceptron import MultilayerPerceptron

#input
_input = [[0, 0], [0,1], [1, 0], [1, 1]]
_expected = [[0], [1], [1], [0]]

learning_rate = 0.5
momentum = 0.8
test_p = 0.25

beta = 0.5

def tanh(x):
    return math.tanh(beta * x)

def tanh_deriv(x):
    return beta * (1 - math.tanh(x)**2)

#activación no lineal y su derivada
def logistic(x):
    return 1 / (1 + math.exp(-2 * beta * x))

def logistic_d(x):
    act = logistic(x)
    #act = x
    return 2 * beta * act * (1 - act)

nn = MultilayerPerceptron(learning_rate, momentum, act_fun=logistic, deriv_fun=logistic_d, split_data=False, test_p=test_p)

nn.entry_layer(2)
nn.add_hidden_layer(5)
nn.output_layer(1)

error = 1
#while error > 0.001:
error = nn.train(_input, _expected, epochs=5000)
#print(error)

for i in range(0, len(_input)):
    print(str(_input[i]) + " -> " + str(nn.predict(_input[i])))