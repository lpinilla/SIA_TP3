import random
import numpy as np

layers = []

class MultilayerPerceptron:

    def __init__(self ):
        global layers

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

    #función para propagar secuencialmente los valores
    def feed_forward(self):
        for i in range(1, len(layers)):
            l = layers[i]
            l["h"] = np.dot(l["weights"], layers[i-1]["v"])
            l["v"] = l["fn"](l["h"])

    #función que propaga regresivamente el valor de error
    def back_propagation(self):
        for i in range(len(layers)-1, 2, -1):
            l = layers[i]
            l_1 = layers[i+1]
            aux = np.dot(l_1["weights"].transpose(), l_1["errors"])
            l["errors"] = [l["deriv"](j) for j in aux]

    def calculate_deltaW(self):
        return 1 #FIXME

    def train(self,inp_data, exp_data):
        #TODO implementar
        print("ok")
