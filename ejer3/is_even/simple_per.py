import random
import math
import numpy as np
from Perceptron import Perceptron
from MultilayerPerceptron import MultilayerPerceptron

#perceptrón simple

learning_rate = 0.01

#activación escalón
def step_activation(x):
    return 1 if x >= 0 else -1

def step_deriv(x):
    return 1

#Ejer 1.1, función AND

_input= [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0,
 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0,
 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

_expected = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

p = Perceptron(2, eta=learning_rate, activation_fun=step_activation, deriv_fun=step_deriv)

iterations, weights, min_weights, curr_error = p.train(_input, _expected)
print("iterations " + str(iterations))
print("weights " + str(weights))

print("Resultados:")
#print(p.guess([-1,-1]), p.guess([-1, 1]), p.guess([1, -1]), p.guess([1, 1]))

#Perceptrón multicapa

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

def arctan(x):
    return math.atan(x)

def arctan_deriv(x):
    y = arctan(x)
    return 1 / (1 + (y ** 2))


nn = MultilayerPerceptron(learning_rate, momentum, act_fun=arctan, deriv_fun=logistic_d, split_data=False, test_p=test_p)

nn.entry_layer(2)
nn.add_hidden_layer(5)
nn.output_layer(1)

error = nn.train(_input, _expected, epochs=5000)

for i in range(0, len(_input)):
    print(str(_input[i]) + " -> " + str(nn.predict(_input[i])))
