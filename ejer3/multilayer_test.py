import numpy as np
import pickle
import random
import math
from MultilayerPerceptron import MultilayerPerceptron

#input
_input = [[-1, -1], [-1,1], [1, -1], [1, 1]]
_expected = [[-1], [-1], [-1], [1]]

learning_rate = 0.1
momentum = 0.9
test_p = 0.25

beta = 0.7 #TODO: ver que valor poner

#activación no lineal y su derivada
def logistic(x):
    return 1 / (1 + np.exp(-2 * beta * x))

def logistic_d(x):
    #act = logistic(x)
    act = x
    return 2 * beta * act * (1 - act)

nn = MultilayerPerceptron(learning_rate, momentum, logistic, logistic_d, test_p)

nn.entry_layer(2)
nn.add_hidden_layer(2)
nn.output_layer(1)

error = nn.train(_input, _expected, epochs=10000)
#print(error)

for i in range(0, len(_input)):
    print(str(_input[i]) + " -> " + str(nn.predict(_input[i])))
