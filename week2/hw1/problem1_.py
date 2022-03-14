prime_list = [2]
for i in range(3,101):
    count = 0
    for p in prime_list:
        if i % p == 0:
            count = count + 1
            break
    if count == 0:
        prime_list.append(i)

print(prime_list)