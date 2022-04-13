from time import time
start = time()
linecount = 0
for i in range(2,101):
    count = 0
    for j in range(2,i):
        if i % j == 0:
            count = count + 1
            break
    if count == 0:
        print(i, end=" ")
print()
print(time() - start)

        # linecount = linecount + 1
        # if linecount % 10 == 0:
        #     print()