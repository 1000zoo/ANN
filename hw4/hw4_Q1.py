PATH = "/Users/1000zoo/Documents/prog/ANN/data_files/hw4_covid/practice_data.txt"

import pickle

with open(PATH, "rb") as f:
    data = pickle.load(f)

print(data)
print(type(data))
print(data.shape)
