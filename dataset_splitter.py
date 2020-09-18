#Archivo para dividir el dataset en distintos batchs
import pickle

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
n = 50
datasets_in = [inputs[i:i+n] for i in range(0, len(inputs), n)][0]
datasets_ex = [expecteds[i:i+n] for i in range(0, len(expecteds), n)][0]

for i in range(0, int(len(inputs) / n)):
    with open(output_basepath + str(i) + ".pickle", "wb") as f:
        pickle.dump(datasets_in, f)
        pickle.dump(datasets_ex, f)
    f.close()
