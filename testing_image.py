res = 1391104 #189696#
verh = 19/13
print(res, verh)

def formula(try_num, verh):
    result = (try_num**2 * verh) * 2
    return result

for i in range(1000):
    bla = formula(i, verh)
    if abs(res-bla) <= 100:
        print(f"found {i}, bla")

print('ddd')

def new_f(result):
    import math
    first = int(math.sqrt(result))
    return result

#aa = new_f(695552)
#print(aa)

for i in range(1000):
    for j in range(1000):
        if i*j*2 == 1391104:
            print(i, j)
