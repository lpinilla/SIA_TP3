import random
import numpy as np

#cantidad de iteraciones maximas para entrenar
max_steps = 1000
#eta adaptativo
#hasta cuantos errores vamos a mirar si variamos eta
error_window = 3
eta_variation = 0.025

class Perceptron:

    #constructor del perceptrón
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

    #función para generar nuevos pesos al azar
    def random_weights(self):
        return np.array(
            [random.random() * 2 - 1 for i in range(1, self.input_size + 2)]
        )

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
        if self.split_data:
            split_idx = int(len(input_arr) * (1 - self.test_p))
            return np.array(inputs[:split_idx]), \
                   expected_arr[:split_idx], \
                   np.array(inputs[split_idx:]), \
                   expected_arr[split_idx:]
        return np.array(inputs), expected_arr, np.array(inputs), expected_arr

    #función para calcular el error
    def calculate_error(self, test_data, expected_values):
        #calcular los valores de excitación (el h)
        exts = np.array([np.dot(self.weights, i) for i in test_data])
        #calcular los valores de activación (el O)
        acts = np.array([self.activation_fun(i) for i in exts])
        error = 0.5 * ((expected_values - acts)**2).sum()
        #eta adaptativo, ir guardando los errores
        self.errors.append(error)
        self.errors = self.errors[1:]
        return error

    #función para calcular el nuevo delta w
    def calculate_deltaW(self, sample_expected, sample, ext):
        return self.eta * (sample_expected - self.activation_fun(ext)) \
               * self.deriv_fun(ext) * sample.transpose()

    #función de activación
    def guess(self, input_arr):
        #agregar un 1 al final del input del usuario, el sesgo
        inp = input_arr
        inp.append(1)
        inp = np.array(inp)
        return self.activation_fun(np.dot(self.weights, inp))

    #función para hacer el eta adaptativo
    def adapt_eta(self, last_error):
        bigger = all(last_error >= i for i in self.errors)
        smaller = all(last_error < i for i in self.errors)
        if bigger:
            self.eta -= eta_variation * self.eta
        if smaller:
            self.eta += eta_variation


    #función de entrenamiento, entrena sobre una tanda de archivos y calcula
    #el error en base a la tanda de prueba
    def train(self, inputs, expected):
        #pasar de una lista a un array de numpy para facilitar cuentas
        inp_data, inp_exp, test_data, test_exp = \
                self.process_input(inputs, expected)
        #incializando valores
        error = 1
        curr_step = 0
        error_min = 1
        min_weights = self.weights
        while error > 0 and curr_step != max_steps:
            #si no encontramos solución en 100 pasos, reiniciar pesos
            #if curr_step % 100:
            #    self.weights = self.random_weights() TODO: ver si no conviene poner los mejores pesos en vez de arrancar random
            #agarramos una muestra al azar del batch con su valor experado
            idx = random.randint(0, len(inp_data) -1)
            sample = inp_data[idx]
            s_exp = inp_exp[idx]
            #calcular el valor de excitación, también conocido como h
            ext = np.dot(self.weights, sample)
            #sumar los nuevos pesos
            self.weights += self.calculate_deltaW(s_exp, sample, ext)
            #calcular el error evaluando todos los datos
            error = self.calculate_error(test_data, test_exp)
            #opcional/experimental: llamar a la función para variar eta
            #en base a cómo es el error en comparación a cómo venía dando
            #self.adapt_eta(error)
            #guardamos el error mínimo y sus pesos para devolverlo más tarde
            if error < error_min:
                error_min = error
                min_weights = self.weights
            curr_step += 1
        return curr_step, self.weights, min_weights, error

