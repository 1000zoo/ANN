PATH = "/Users/1000zoo/Documents/prog/ANN/data_files/hw4_covid/practice_data.txt"

import pickle
import numpy as np
import copy

with open(PATH, "rb") as f:
    data = pickle.load(f)

train_data = data[:350]
test_data = data[350:]

def generator(data, time_steps=10):
    batch = len(data) - time_steps
    input = []
    target = []

    for i in range(batch):
        try:
            input.append(data[i:time_steps+i])
            target.append(data[time_steps+i])
        except IndexError as err:
            print(err)
    
    input = np.array(input)
    target = np.array(target)
    input, target = normalize(input, target)
    return input, target

def normalize(data, target):
    norm_data = copy.deepcopy(data)
    for i, d in enumerate(data):
        max_value = np.max(d)
        if max_value <= 0:
            continue
        norm_data[i] /= max_value
        target[i] /= max_value
    return norm_data, target

train_input, train_target = generator(train_data)
test_input, test_target = generator(test_data)

print(train_input.shape, train_target.shape)
print(test_input.shape, test_target.shape)

with open("train_input.txt", 'w') as f:
    line = "["

    for data in train_input:
        for d in data:
            line += str(d)
            line += ", \n"
        line += "], "
    line += "]\n"
    f.write(line)