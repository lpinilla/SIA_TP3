import numpy as np
import pickle
import random
import math
from MultilayerPerceptron import MultilayerPerceptron
#cargar data
f = open('ej3_resources/numbers_as_array.pickle', 'rb')
_input = pickle.load(f)
_expected = pickle.load(f)
f.close()

learning_rate = 2.3
momentum = 0.9
test_percentage = 0.1

#definimos la función de activación no lineal entre 0 y 1 (porque estamos buscando probabilidades)

beta = 0.005 #TODO: ver que valor poner
#activación no lineal y su derivada
def logistic(x):
    return 1 / (1 + np.exp(-2 * beta * x))

def logistic_deriv(x):
    act = logistic(x)
    #act = x
    return 2 * beta * act * (1 - act)

def tanh(x):
    return math.tanh(beta * x)

def tanh_deriv(x):
    return beta * (1 - tanh(x) ** 2)

def arctan(x):
    return math.atan(x)

def arctan_deriv(x):
    y = arctan(x)
    return 1 / (1 + (y ** 2))

#Creamos la red
nn = MultilayerPerceptron(learning_rate, momentum, act_fun=logistic, deriv_fun=logistic_deriv, test_p=test_percentage, use_momentum=True)

#Definimos el modelo de la red

#vamos a tener 1 entrada por byte, por lo que son 7 x 5 = 35 entradas
#como tenemos que decir si el dígito está entre 0 y 9, vamos a tener
#10 nodos de salida en donde indicamos la probabilidad de que sea ese número
nn.entry_layer(35)
#capa oculta intermedia
nn.add_hidden_layer(70)
#la salida
nn.output_layer(10)

error = 1

error = nn.train(_input, _expected, epochs=1000)
print(error)


idx = 1
print(str(idx) + " -> " + str(nn.predict(_input[idx])))

