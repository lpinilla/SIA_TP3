import random
import numpy as np
import pickle
from Perceptron import Perceptron

learning_rate = 0.001

#FIXME: la normalización llevó a los números negativos al rango [0,1]. Revisar si esto está bien

#Ejercicio 2

#donde se encuentran los datasets
datasets_basepath = "ej2_resources/"

#activación lineal
def linear_activation(x):
    return x

def linear_deriv(x):
    return 1

beta = 1 #TODO: ver que valor poner
#activación no lineal y su derivada
def no_linear_activation(x):
    return 1 / (1 + np.exp(-2 * beta * x))

def no_linear_deriv(x):
    act = no_linear_activation(x)
    return 2 * beta * act * (1 - act)

#Creamos perceptron lineal que dividiendo el dataset en 10% para testeo
p = Perceptron(3, learning_rate, activation_fun=linear_activation, deriv_fun=linear_deriv, split_data=True, test_p=0.1)

#Creamos perceptron no-lineal que dividiendo el dataset en 10% para testeo
#p = Perceptron(3, learning_rate, activation_fun=no_linear_activation, deriv_fun=no_linear_deriv, split_data=True, test_p=0.1)

error = 5
#indicar en cuantas partes está dividido el dataset
n_of_parts = 4

while error > 0.0000001:
    #Agarramos un dataset random
    #data normalizada
    f = open(datasets_basepath + "n_" + str(random.randint(0,n_of_parts - 1)) + ".pickle", "rb")
    #data no normalizada
    #f = open(datasets_basepath + str(random.randint(0,n_of_parts - 1)) + ".pickle", "rb")
    #cargar los valores de entrada
    inp = pickle.load(f)
    #cargar los valores esperados
    exp = pickle.load(f)
    f.close()
    #entrenar con este dataset
    iterations, weights, min_weights, error = p.train(inp, exp)
    #p.weights = min_weights
    print(error)

#Testear al perceptrón
#f = open(datasets_basepath + str(random.randint(0,n_of_parts - 1)) + ".pickle", "rb")
f = open(datasets_basepath + "n_" + str(random.randint(0,n_of_parts - 1)) + ".pickle", "rb")
inp = pickle.load(f)
exp = pickle.load(f)
f.close()
for i in range(0, len(inp)):
    print("Expected " + str(exp[i]))
    print("Result " + str(p.guess(inp[i])))
