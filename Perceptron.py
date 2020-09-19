import random
import numpy as np

max_steps = 1000
error_window = 4
eta_variation = 0.001

class Perceptron:

    def __init__(self, input_size, eta, \
                 activation_fun, deriv_fun, split_data=False, test_p=1):
        global max_steps
        global error_window
        global eta_variation
        self.input_size = input_size
        self.weights = self.random_weights()
        self.eta = eta
        self.activation_fun = activation_fun
        self.deriv_fun = deriv_fun
        self.split_data = split_data
        self.test_p = test_p
        self.errors = [10000 for i in range(0, error_window)]

    def random_weights(self):
        return np.array(
            [random.random() * 2 - 1 for i in range(1, self.input_size + 2)]
        )

    #agregar un 1 al valor y devolver un numpy array
    def process_input(self, input_arr, expected_arr):
        #checkear que la cantidad de inputs sea la misma que
        #la cantidad de pesos
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
        if self.split_data:
            split_idx = int(len(input_arr) * (1 - self.test_p))
            return np.array(inputs[:split_idx]), \
                   expected_arr[:split_idx], \
                   np.array(inputs[split_idx:]), \
                   expected_arr[split_idx:]
        return np.array(inputs), expected_arr, np.array(inputs), expected_arr

    def calculate_error(self, test_data, expected_values):
        exts = np.array([np.dot(self.weights, i) for i in test_data])
        acts = np.array([self.activation_fun(i) for i in exts])
        error = 0.5 * ((expected_values - acts)**2).sum()
        self.errors.append(error)
        self.errors = self.errors[1:]
        return error

    def calculate_deltaW(self, sample_expected, sample, ext):
        return self.eta * (sample_expected - self.activation_fun(ext)) \
               * self.deriv_fun(ext) * sample.transpose()

    #función de activación
    def guess(self, input_arr):
        inp = input_arr
        inp.append(1)
        inp = np.array(inp)
        return self.activation_fun(np.dot(self.weights, inp))

    def adapt_eta(self, last_error):
        bigger = all(last_error >= i for i in self.errors)
        smaller = all(last_error < i for i in self.errors)
        if bigger:
            self.eta -= eta_variation * self.eta
        if smaller:
            self.eta += eta_variation


    def train(self, inputs, expected):
        #pasar de una lista a un array de numpy
        inp_data, inp_exp, test_data, test_exp = \
                self.process_input(inputs, expected)
        error = 1
        curr_step = 0
        error_min = 1
        min_weights = self.weights
        while error > 0 and curr_step != max_steps:
            #si no encontramos solución en 100 pasos, reiniciar pesos
            if curr_step % 1000:
                self.weights = self.random_weights()
            #agarramos una muestra al azar con su valor experado
            idx = random.randint(0, len(inp_data) -1)
            sample = inp_data[idx]
            s_exp = inp_exp[idx]
            #calcular el valor de excitación, también conocido como h
            ext = np.dot(self.weights, sample)
            #sumar los nuevos pesos
            self.weights += self.calculate_deltaW(s_exp, sample, ext)
            #calcular el error evaluando todos los datos
            error = self.calculate_error(test_data, test_exp)
            self.adapt_eta(error)
            if error < error_min:
                error_min = error
                min_weights = self.weights
            curr_step += 1
        return curr_step, self.weights, min_weights, error

