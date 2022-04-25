import os

from numpy import character
file_name = os.path.basename(__file__)
ch = ".py"

for c in ch:
    file_name = file_name.replace(c, "")

wlist = file_name.split("_")
print(wlist[-1])