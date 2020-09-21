import numpy as np
import pickle
import random
from MultilayerPerceptron import MultilayerPerceptron

#cargar data
f = open('ej3_resources/numbers_as_array.pickle', 'rb')
_input = pickle.load(f)
_expected = pickle.load(f)
f.close()

learning_rate = 0.01
momentum = 0.9
test_percentage = 0.1

#definimos la función de activación no lineal entre 0 y 1 (porque estamos buscando probabilidades)

beta = 0.7 #TODO: ver que valor poner
#activación no lineal y su derivada
def no_linear_activation(x):
    return 1 / (1 + np.exp(-2 * beta * x))

def no_linear_deriv(x):
    act = no_linear_activation(x)
    return 2 * beta * act * (1 - act)


#Creamos la red
nn = MultilayerPerceptron(learning_rate, momentum, no_linear_activation, no_linear_deriv, test_percentage)

#Definimos el modelo de la red

#vamos a tener 1 entrada por byte, por lo que son 7 x 5 = 35 entradas
#como tenemos que decir si el dígito está entre 0 y 9, vamos a tener
#10 nodos de salida en donde indicamos la probabilidad de que sea ese número
nn.add_layer(35)
#capa oculta intermedia
nn.add_layer(35)
#la salida
nn.add_layer(9)

error = 1

while error > 0.0001:
    #elegir input al azar
    #idx = random.randint(0, len(_input) - 1)
    #_in = _input[idx]
    #_ex = _expected[idx]
    error = nn.train(_input, _expected)
    print(error)

print(nn.guess(_input[1]))
