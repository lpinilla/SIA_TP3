import random
import numpy as np

class Perceptron:

    def __init__(self, inputSize, learning_rate, \
                 activation_function, deriv_fun):
        self.inputLayerSize = inputSize
        self.weights = np.array([random.random() * 2 - 1 for i in range(1, inputSize + 2)])
        self.eta = learning_rate
        self.activation_function = activation_function
        self.deriv_fun = deriv_fun

    #agregar un 1 al valor y devolver un numpy array
    def process_input(self, input_arr):
        #checkear que la cantidad de inputs sea la misma que la cantidad de pesos
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
        return np.array(inputs)

    def calculate_error(self, test_data, expected_values):
        return (0.5 * (expected_values - \
            np.array([self.activation_function(i) for i in test_data]))**2)\
            .sum()

    def calculate_deltaW(self, sample_expected, sample, ext):
        return self.eta * (sample_expected - self.activation_fun(ext)) \
               * self.deriv_fun(ext) *  sample.transpose()


    def train(self, inputs_arr, expected):
        #pasar de una lista a un array de numpy
        inputs_data = self.process_input(inputs_arr)
        error = 1
        curr_step = 0
        error_min = 1
        max_steps = 100000
        while error > 0 and curr_step != max_steps:
            #si no encontramos solución en 100 pasos, reiniciar pesos
            if curr_step % 100:
                self.weights = np.array([
                    random.random() * 2 - 1 for i in \
                    range(1, len(self.weights) + 1)
                ])
            #agarramos una muestra al azar con su valor experado
            idx = random.randint(0, len(inputs_data) -1)
            sample = inputs_data[idx]
            s_exp = expected[idx]
            #calcular el valor de excitación, también conocido como h
            ext = np.dot(self.weights, sample)
            #sumar los nuevos pesos
            self.weights += self.calculate_deltaW(s_exp, sample, ext)
            #calcular el error evaluando todos los datos
            error = self.calculate_error(inputs_data, expected)
            if error < error_min:
                error_min = error
            curr_step += 1
        return curr_step, self.weights, error

