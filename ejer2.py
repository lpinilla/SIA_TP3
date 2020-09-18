import random
import numpy as np
import pickle
from Perceptron import Perceptron

learning_rate = 0.2

#Ejercicio 2

datasets_basepath = "ej2_resources/"

#activación lineal
def linear_activation(x):
    return x

def linear_deriv(x):
    return 1

#Creamos perceptron lineal que dividiendo el dataset en 10% para testeo
p3 = Perceptron(3, learning_rate, activation_fun=linear_activation, deriv_fun=linear_deriv, split_data=True, test_p=0.1)

error = 5

while error > 0:
    #Agarramos un dataset random
    f = open(datasets_basepath + str(random.randint(0,9)) + ".pickle", "rb")
    inp = pickle.load(f)
    exp = pickle.load(f)
    f.close()
    iterations, weights, error = \
        p3.train(inp, exp)

f = open(datasets_basepath + str(random.randint(0,9)) + ".pickle", "rb")
inp = pickle.load(f)
exp = pickle.load(f)

for i in range(0, len(inp)):
    print("Expected " + str(exp[i]))
    print("Result " + str(p3.guess(inp[i])))

#activacion no lineal
#beta = 0.7 #TODO: ver que valor poner
#
#def no_linear_activation(x):
#    return 1 / (1 + np.exp(-2 * beta * x))
#
#def no_linear_deriv(x):
#    act = no_linear_activation(x)
#    return 2 * beta * act * (1 - act)
