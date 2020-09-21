#Archivo para dividir el dataset en distintos batchs
import pickle
import numpy as np

input_data= "ejer2_conj_entrenamiento.txt"
expected_data = "ejer2_salida_deseada.txt"
output_basepath = "./ej2_resources/"

#cargar los inputs
inputs = []
with open(input_data) as f:
    for line in f:
        inputs.append([float(i) for i in line[:-1].split("   ")[1:]])
f.close()

#Cargar los valores esperados
expecteds = []
with open(expected_data) as f:
    for line in f:
        expecteds.append(float(line[:-1].split("   ")[1]))
f.close()

#separar en distintas sublistas el dataset
#n indica el número de datos de cada particion
n = 64
datasets_in = [inputs[i:i+n] for i in range(0, len(inputs), n)]
datasets_ex = [expecteds[i:i+n] for i in range(0, len(expecteds), n)]

def normalize(arr):
    arr = np.array(arr)
    arr = (arr - np.min(arr)) / np.ptp(arr)
    return arr.tolist()

for i in range(0, len(datasets_in)):
#    datasets_in[i] = normalize(datasets_in[i])
    datasets_ex[i] = normalize(datasets_ex[i])

for i in range(0, len(datasets_in)):
    with open(output_basepath + "n_" + str(i) + ".pickle", "wb") as f:
        pickle.dump(datasets_in[i], f)
        pickle.dump(datasets_ex[i], f)
    f.close()
