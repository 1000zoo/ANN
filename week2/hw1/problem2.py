import numpy as np

data = np.loadtxt('hw1.csv', delimiter=',')
print("sum axis = 1 (행의 합): ")
print(np.sum(data, axis=1))
print("sum axis = 0 (열의 합): ")
print(np.sum(data, axis=0))
