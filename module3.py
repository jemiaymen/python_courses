
# %%
i = 0
while (i < 100):
    print(i)
    i += 10  # i = i + 10

# %%


def somme(tab):
    result, i = 0, 0
    size = len(tab)
    while (i < size):
        result += tab[i]
        i += 1

    return result


print(somme([5, 9, 10, 5, 4]))

# %%
mat = [[2, 5, 3], [8, 7, 9], [2, 1]]
i, j, result = 0, 0, 0
size = len(mat)
while (i < size):
    j = 0
    size2 = len(mat[i])
    while (j < size2):
        result += mat[i][j]
        j += 1
    i += 1

print(result)

# %%

mat = [[2, 5, 3], [8, 7, 9], [2, 1]]
result = 0
for x in mat:
    for y in x:
        result += y

print(result)

# %%
x = range(3, 20, 2)
for n in x:
  print(n)
# %%
try:
    open('filenotfound')
except ZeroDivisionError:
    print("Division error")
except ValueError:
    print("is not int")
finally:
    print('not devided and not value error ')


# %%
