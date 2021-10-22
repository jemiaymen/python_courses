#%%
import numpy as np
a = np.zeros((2,3),dtype=np.int32)
# %%
a
# %%
import numpy as np
a = np.array([[[1, 5, 3], [2, 9, 7]], [[11, 55, 33], [22, 99, 77]]])

b = np.array([[[111, 555, 333], [222, 999, 777]], [[1, 5, 3], [2, 9, 7]]])

c = np.concatenate((a,b),axis=1)
# %%
print(c)
# %%
a = np.array([1 , 2 ])
b = np.array([3 , 4])

print(np.concatenate((a,b),axis=0))
# %%
a = np.array([[1],[2]])
b = np.array([[3],[4]])

print(np.concatenate((a,b),axis=1))
# %%
a = np.array([[1 , 2],[3,4]])
b = np.array([[5,6]])

print(np.concatenate((a,b)))
# %%
a = np.arange(11)
condition = (a % 2 == 0) | (a != 7)
print(a[condition])
# %%
a=np.array([1,2,3])
b = np.ones(3)

print(a - b)
# %%
print(a / 0)
# %%
a = np.arange(1,100,1.2)
# %%
a.size

# %%
a.std()
# %%
a.prod()
# %%
a = np.array([1, 2 , 4 , 1 , 5 , 4 , 1 , 2])
v , index = np.unique(a,return_index=True)
# %%
print(v)
print(index)
# %%
v , count = np.unique(a , return_counts=True)
print(v)
print(count)

# %%
help(np)
# %%
help(np.mean)
# %%
np.median