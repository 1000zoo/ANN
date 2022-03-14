import numpy as np

#ex1
print("ex1")
arr = [[1,2,3,4], [5,6,7,8], [11,12,13,14]]
ndarr = np.array(arr)
print(ndarr)

#ex2
print("ex2")
print(ndarr[1][2])

#ex3
print("ex3")
sndarr = ndarr[(0,2), :]
print(sndarr, type(sndarr))
print("="*15)
print(ndarr[0::2])