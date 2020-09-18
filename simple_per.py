import random
import numpy as np
from Perceptron import Perceptron

learning_rate = 0.2

#activación escalón
def step_activation(x):
    return 1 if x >= 0 else -1

def step_deriv(x):
    return 1

#Ejer 1.1, función AND

and_input_data= [[-1,-1], [-1,1], [1,-1], [1,1]]
and_expected_values = [-1, -1, -1, 1]

p = Perceptron(2, eta=learning_rate, activation_fun=step_activation, deriv_fun=step_deriv)

iterations, weights, curr_error = p.train(and_input_data, and_expected_values)
print("iterations " + str(iterations))
print("weights " + str(weights))

print("Resultados:")
print(p.guess([-1,-1, 1]), p.guess([-1, 1, 1]), \
      p.guess([1, -1, 1]), p.guess([1, 1, 1]))

#Ejer 1.2, función XOR

#La función no es linealmente separable por lo que esperamos
#que el programa corte por límite de pasos

p2 = Perceptron(2, eta=learning_rate, activation_fun=step_activation, deriv_fun=step_deriv)


xor_input_data= [[-1,-1], [-1,1], [1,-1], [1,1]]
xor_expected_values = [-1, 1, 1, -1]

iterations, weights, curr_error = \
    p2.train(xor_input_data, xor_expected_values)

print("iterations " + str(iterations))
print("weights " + str(weights))

print("Resultados:")
print(p2.guess([-1,-1, 1]), p2.guess([-1, 1, 1]), \
      p2.guess([1, -1, 1]), p2.guess([1, 1, 1]))
