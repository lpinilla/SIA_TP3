import random
from math import log
import numpy as np

layers = []
max_steps = 1000

class MultilayerPerceptron:

    def __init__(self, eta, momentum):
        global layers
        global max_steps
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
            "e": self.random_array(n_of_nodes),
            #función de activación
            "fn": act_fun,
            #derivada de la función de activación
            "deriv": deriv_fun
        }
        layers.append(layer)

    def setup_entries(self, entries):
        entry = layers[0]
        entry["v"] = entries
        entry["w"] = []
        entry["prev_w"] = []
        entry["h"] = []
        entry["e"] = []

    #agregar un 1 al valor (para el sesgo) y devolver un numpy array
    def process_input(self, input_arr, expected_arr):
        #checkear que #inputs sea igual a #pesos
        for inp in input_arr:
            assert len(self.weights) == len(inp) + 1
        inputs = []
        for inp in input_arr:
            #sumar el 1 como input para el sesgo
            input_bias = np.copy(inp)
            #agregar el 1 para el sesgo
            input_bias = np.append(input_bias, 1)
            #convertir en array de numpy
            input_data = np.array(input_bias)
            inputs.append(input_data)
        #si se seteo, partir el dataset en input y test data
        #en base al % introducido
        split_idx = int(len(input_arr) * (1 - self.test_p))
        #setear la capa 0
        layer_0 = np.array(inputs[:split_idx])
        setup_entries(layer_0)
        return layer_0, expected_arr[:split_idx], \
               np.array(inputs[split_idx:]), expected_arr[split_idx:]

    def guess(self, _input):
        _data = _input
        _data.append(1)
        data = np.array(_data)
        self.setup_entries(data)
        self.feed_forward()
        return layers[len(layers)-1]["v"]

    def calculate_error(self, test_data, expected):
        guesses = [guess(i) for i in test_data]
        aux = 0
        for i in range(0, len(expected)):
            exp = expected[i]
            guess = guesses[i]
            for j in range(0, len(exp)):
                e_1 = 1 + exp[j]
                e__1 = 1 - exp[j]
                o_1 = 1 + guess[j]
                o__1 = 1 - guess[j]
                aux += (e_1 * log(e_1 / o_1) + e__1 * log(e__1, o__1))
        return 0.5 * aux

    #función para propagar secuencialmente los valores
    def feed_forward(self):
        for i in range(1, len(layers)):
            l = layers[i]
            l["h"] = np.dot(l["w"], layers[i-1]["v"])
            l["v"] = l["fn"](l["h"])

    #función que propaga regresivamente el valor de error
    def back_propagation(self, inputs):
        for i in range(len(layers)-1, 2, -1):
            l = layers[i]
            l_1 = layers[i+1]
            aux = np.dot(l_1["w"].transpose(), l_1["e"])
            l["e"] = [l["deriv"](j) for j in aux]
            #actualizar los pesos
            w = l["prev_w"]
            l["w"] += self.eta * \
                np.dot(l["e"].transpose(), layers[i-1]["v"]) + momentum * w
        l = layers[1]
        w = l["prev_w"]
        l["w"] += self.eta * np.dot(l[2]["e"], inputs) + momentum * w

    def train(self, inputs, expected):
        inp_data, inp_exp, test_data, test_exp = \
            self.process_input(inputs, expected)
        error = 1
        curr_step = 0
        error_min = 1
        while error > 0 and curr_step != max_steps:
            #agarrar un índice random para agarrar una muestra
            idx = random.randint(0, len(inp_data) - 1)
            _in = inp_data[idx]
            _ex = inp_exp[idx]
            #setear las entradas en base a la muestra
            self.setup_entries(_in)
            #hacer feed forward
            self.feed_forward()
            #hacer back propagation del error
            self.back_propagation(_in)
            #calcular el error
            error = self.calculate_error(test_data, test_exp)
            if error < error_min:
                error_min = error
            curr_step += 1
        return curr_step, layers, error
