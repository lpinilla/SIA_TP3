import random
from math import log
import numpy as np

layers = []
max_steps = 1000

class MultilayerPerceptron:

    def __init__(self, eta, momentum, act_fun, deriv_fun, test_p):
        global layers
        global max_steps
        self.eta = eta
        self.momentum = momentum
        self.act_fun = act_fun
        self.deriv_fun = deriv_fun
        self.test_p = test_p

    def print_layer(self, i):
        l = layers[i]
        print("i: " + str(i))
        print(" w: " + str(l["w"]))
        print(" v: " + str(l["v"]))
        print(" h: " + str(l["h"]))
        print(" e: " + str(l["e"]))

    def print_layers(self):
        for i in range(0, len(layers)):
            self.print_layer(i)

    def create_layer(self, n_of_nodes, fn=None, d_fn=None):
        layer = {
            #pesos de cada nodo o entradas si es la capa inicial
            "w" : np.ones(n_of_nodes) if not layers \
            else [np.ones(len(layers[-1]["v"])) for i in range(0, n_of_nodes)],
            #pesos anteriores, para usar momentum
            "prev_w" : np.zeros(n_of_nodes) if not layers \
            else [np.zeros(len(layers[-1]["v"])) for i in range(0, n_of_nodes)],
            #valores de activación
            "v" : np.ones(n_of_nodes),
            #valores de exitación
            "h" : np.ones(n_of_nodes),
            #valores de error
            "e": np.ones(n_of_nodes),
            #función de activación
            "fn": fn if fn != None else self.act_fun,
            #derivada de la función de activación
            "deriv": d_fn if d_fn != None else self.deriv_fun
        }
        return layer

    def entry_layer(self, n_of_nodes, fn=None, deriv=None):
        l = self.create_layer(n_of_nodes+1, fn=fn, d_fn=deriv)
        layers.append(l)

    def add_hidden_layer(self, n_of_nodes, act_fun=None, deriv_fun=None):
        l = self.create_layer(n_of_nodes+1, fn=act_fun, d_fn=deriv_fun)
        layers.append(l)

    def output_layer(self, n_of_nodes, fn=None, deriv=None):
        l = self.create_layer(n_of_nodes, fn=fn, d_fn=deriv)
        layers.append(l)

    def setup_entries(self, entries):
        entry = layers[0]
        entry["v"] = entries
        entry["w"] = []
        entry["prev_w"] = []
        entry["h"] = []
        entry["e"] = []

    #agregar un 1 al valor (para el sesgo) y devolver un numpy array
    def process_input(self, input_arr, expected_arr):
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
        return np.array(inputs[:split_idx]), expected_arr[:split_idx], \
               np.array(inputs[split_idx:]), expected_arr[split_idx:]

    def predict(self, _input):
        return self.guess(np.append(np.array(_input), 1))

    def guess(self, _input):
        self.setup_entries(_input)
        self.feed_forward()
        return layers[len(layers)-1]["v"]

    def calculate_error(self, test_data, test_exp):
        aux = 0
        guesses = [self.guess(i) for i in test_data]
        for i in range(len(test_exp)):
            _ex = test_exp[i]
            guess = guesses[i]
            for j in range(0, len(_ex)):
                aux += (_ex[j] - guess[j]) ** 2
        return 0.5 * aux

    #def calculate_error(self, test_data, expected):
    #    guesses = [self.guess(i) for i in test_data]
    #    aux = 0
    #    for i in range(0, len(expected)):
    #        exp = expected[i]
    #        guess = guesses[i]
    #        for j in range(0, len(exp)):
    #            e_1 = 1 + exp[j]
    #            e__1 = 1 - exp[j]
    #            o_1 = 1 + guess[j]
    #            o__1 = 1 - guess[j]
    #            aux += (e_1 * log(e_1 / o_1) + e__1 * log(e__1, o__1))
    #    return 0.5 * aux

    #función para propagar secuencialmente los valores
    def feed_forward(self):
        for i in range(1, len(layers)):
            l = layers[i]
            h = [np.dot(l["w"][j], layers[i-1]["v"]) for j in range(0, len(l["h"]))]
            l["h"] = np.array(h)
            l["v"] = np.array([l["fn"](i) for i in l["h"]])

    #función que propaga regresivamente el valor de error
    def back_propagation(self):
        for i in range(len(layers) - 1, 1, -1):
            l = layers[i]
            l_1 = layers[i-1]
            #calculamos los nuevos errores en base a los de la capa superior
            errors = []
            for j in range(0, len(l_1["e"])):
                #agarrar todas las conexiones del nodo j con la capa superior
                w_1 = np.array([l["w"][k][j] for k in range(0, len(l["w"]))])
                #calcular los producto punto entre pesos y
                #errores de la capa superior
                aux = np.dot(w_1, l["e"])
                errors.append(l_1["deriv"](l_1["h"][j]) * aux)
            l_1["e"] = errors


    def update_weights(self):
        for i in range(len(layers) - 1, 1, -1):
            l = layers[i]
            l_1 = layers[i-1]
            w = l["w"]
            delta_w = 0
            for e in range(0, len(l["e"])):
                for j in range(0, len(w)):
                    aux = l["e"][e] * l_1["v"][j]
                    #delta_w = np.multiply(self.eta * l["deriv"](l["h"][j]) ,  aux)
                    delta_w = self.eta * l["deriv"](l["h"][j]) +  aux
                    #print(l["prev_w"][j])
                    l["w"][j] = np.add(l["w"][j], delta_w)# + self.momentum * l["prev_w"][j]
                    l["prev_w"][j] = delta_w


    def calculate_last_layer_error(self, expected):
        l = layers[-1]
        l["e"] = [l["deriv"](l["h"][i]) * (expected[i] - l["v"][i]) \
                  for i in range(0, len(l["e"]))]


    def train(self, inputs, expected, epochs):
        inp_data, inp_exp, test_data, test_exp = \
            self.process_input(inputs, expected)
        error = 1
        curr_step = 0
        error_min = 1
        idxs = [i for i in range(0, len(inp_data))]
        for i in range(0, epochs):
            order = random.sample(idxs, len(idxs))
            for j in range(0, len(order)):
                #agarrar un índice random para agarrar una muestra
                idx = order[j]
                _in = inp_data[idx]
                _ex = inp_exp[idx]
                self.setup_entries(_in)
                #hacer feed forward
                self.feed_forward()
                #calcular el delta error de la última capa
                self.calculate_last_layer_error(_ex)
                #retropropagar el error hacia las demás capas
                self.back_propagation()
                #ajustar los pesos
                self.update_weights()
                #calcular el error
                error = self.calculate_error(test_data, test_exp)
                if error < error_min:
                    error_min = error
                #curr_step += 1
        print(error)
        return error
