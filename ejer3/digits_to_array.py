import pickle

numbers = []
idx = 0
number = []

with open('numbers_pixelmaps.txt', 'r') as f:
    for line in f:
        if idx % 7 == 0:
            numbers.append(number)
            number = []
        row = [int(i) for i  in line[:-1].split(' ')]
        number.extend(row)
        idx += 1
    numbers.append(number)
#saltear el primero que está vacío
numbers = numbers[1:]
f.close()

#armar los valores esperados que van a ser arrays en donde indicamos qué valor es. Ejemplo, si la imagen es de un 3, el valore esperado es [0 0 0 1 0 0 0 0 0 0], ya que queremos uqe la probabilidade que detecte un 3 sea 1 y las demás probabilidades que sean 0

expecteds = []

for i in range(0, 10):
    _exp = [1 if j == i else 0 for j in range(0, 10)]
    expecteds.append(_exp)

f = open('ej3_resources/numbers_as_array.pickle', 'wb')
pickle.dump(numbers, f)
pickle.dump(expecteds, f)
