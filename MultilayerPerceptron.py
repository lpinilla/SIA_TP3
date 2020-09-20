import random
from math import log
import numpy as np

layers = []

class MultilayerPerceptron:

    def __init__(self, eta, momentum):
        global layers
        self.eta = eta
        self.momentum = momentum

    #función para generar un array con valores al azar
    #introducimos un valor de más para el sesgo
    def random_array(self, n):
        return np.array(
            [random.random() * 2 - 1 for i in range(1, n + 2)]
        )

    #Agregamos una capa a la red
    def add_layer(self, n_of_nodes, act_fun, deriv_fun):
        layer = {
            #pesos
            "w" : self.random_array(n_of_nodes),
            #pesos anteriores, para usar momentum
            "prev_w" : self.random_array(n_of_nodes),
            #valores de activación
            "v" : self.random_array(n_of_nodes),
            #valores de exitación
            "h" : self.random_array(n_of_nodes),
            #valores de error
            "errors": self.random_array(n_of_nodes),
            #función de activación
            "fn": act_fun,
            #derivada de la función de activación
            "deriv": deriv_fun
        }
        layers.append(layer)

    def guess(self, _input):
        _data = _input
        _data.append(1)
        data = np.array(_data)
        layers[0]["v"] = data
        feed_forward(self)
        return layers[len(layers)-1]["v"]

    def calculate_error(self, test_data, expected):
        guesses = [guess(i) for i in test_data]
        aux = 0
        for i in range(0, len(expected)):
            exp = expected[i]
            for j in range(0, len(exp)):
                e_1 = 1 + exp[j]
                e__1 = 1 - exp[j]
                o_1 = 1 + guesses[i][j]
                o__1 = 1 - guesses[i][j]
                aux += (e_1 * log(e_1 / o_1) + e__1 * log(e__1, o__1))
        return 0.5 * aux


    #función para propagar secuencialmente los valores
    def feed_forward(self):
        for i in range(1, len(layers)):
            l = layers[i]
            l["h"] = np.dot(l["weights"], layers[i-1]["v"])
            l["v"] = l["fn"](l["h"])

    #función que propaga regresivamente el valor de error
    def back_propagation(self, inputs):
        for i in range(len(layers)-1, 2, -1):
            l = layers[i]
            l_1 = layers[i+1]
            aux = np.dot(l_1["weights"].transpose(), l_1["errors"])
            l["errors"] = [l["deriv"](j) for j in aux]
            #actualizar los pesos
            w = l["prev_w"]
            l["weights"] += self.eta * \
                np.dot(l["errors"].transpose(), layers[i-1]["v"]) + momentum * w
        l = layers[1]
        w = l["prev_w"]
        l["weights"] += self.eta * np.dot(l[2]["errors"], inputs) + momentum * w

    def train(self,inp_data, exp_data):
        #TODO inyectar datos en la capa 0
        #TODO implementar
        print("ok")
