# Module 5 Numpy

NumPy (Numerical Python) is an open source Python library that’s used in almost every field of science and engineering. It’s the universal standard for working with numerical data in Python, and it’s at the core of the scientific Python and PyData ecosystems

## installing numpy

To install NumPy, we strongly recommend using a scientific Python distribution. If you’re looking for the full instructions for installing NumPy on your operating system, see Installing NumPy.


### with pip 

``` terminal
pip install numpy
```

### with anaconda

``` terminal
conda install numpy
```

## use numpy

To access NumPy and its functions import it in your Python code like this:

``` python
import numpy as np
```
We shorten the imported name to np for better readability of code using NumPy.


### What’s the difference between a Python list and a NumPy array?

NumPy gives you an enormous range of fast and efficient ways of creating arrays and manipulating numerical data inside them. While a Python list can contain different data types within a single list, all of the elements in a NumPy array should be homogeneous. The mathematical operations that are meant to be performed on arrays would be extremely inefficient if the arrays weren’t homogeneous.

### Why use Numpy ?

NumPy arrays are faster and more compact than Python lists. An array consumes less memory and is convenient to use. NumPy uses much less memory to store data and it provides a mechanism of specifying the data types. This allows the code to be optimized even further.

## numpy array

One way we can initialize NumPy arrays is from Python lists, using nested lists for two- or higher-dimensional data.

``` python
a = np.array([1 , 3 , 8 , 9])

# or

a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

print(a[0])

# >>> array([1 2 3 4])

```

### create a basic array


Besides creating an array from a sequence of elements, you can easily create an array filled with 0’s:


``` python
a = np.zeros(2)

print(a)

# >>> array([0., 0.])

# or an array filled with 1's

b = np.ones(2)

print(b)

# >>> array([1. , 1.])

# or an empty with random values

c = np.empty(2)

print(c)

# >>> array([0. , 0.])

# empty is faster more then zeros() or ones()

# create range element with numpy

r = np.arange(2, 9, 1.2)

print(r)

# >>> array([2. , 3.2, 4.4, 5.6, 6.8, 8. ])

# not like range we can step with float numbers

# create array with values that are spaced linearly in specified interval

lin  = np.linspace(1, 20 , num=3)

print(lin)

# >>> array([ 1. , 10.5, 20. ])

lin  = np.linspace(1, 20 , num=10)

print(lin)

# >>> array([ 1., 3.11111111, 5.22222222, 7.33333333, 9.44444444, 11.55555556, 13.66666667, 15.77777778, 17.88888889, 20.])

# the default data type is np.float64, but you can explicitly specify which data type you want with dtype

x = np.ones(10,dtype=np.int32)

print(x)

# >>> array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

``` 


### adding, removing and sorting elements

Sorting an element is simple with np.sort(). You can specify the axis, kind, and order when you call the function.

``` python

array = np.array([8,5,7,9,4,4,6,2,3])

print(np.sort(array))

# >>> array([2 3 4 4 5 6 7 8 9])

# You can concatenate two or more array with np.concatenate().

a = np.array([1,3,5,7])
b = np.array([2,4,6,8])

c = np.concatenate((a,b))

print(c)

# >>> array([1, 3, 5, 7, 2, 4, 6, 8])

# you can concatenate with axis

a = np.array([[1,2],[3,4]])

b = np.array([[5,6]])

c = np.concatenate((a,b) , axis=0)

print(c)

# >>> array([[1, 2], [3, 4], [5, 6]])

# size of array is the number of element

np_array = np.array([[1 ,2, 3] , [5 ,8,9]])

print(np_array.size)

# >>> 6

# dimension of array

print(np_array.ndim)

# >>> 2

# shape of array

print(np_array.shape)

# >>> (2 , 3)

```


You can reshape an array by using array.reshape() without changing the data.

Note : remember that when you use the reshape method, the array you want to produce needs to have the same number of elements as the original array. If you start with an array with 12 elements, you’ll need to make sure that your new array also has a total of 12 elements.


``` python

a = np.arange(10)

print(a)

# >>> array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

b = a.reshape(5,2)

print(b)

""" >>> array([[0, 1],
               [2, 3],
               [4, 5],
               [6, 7],
               [8, 9]])
"""

# for optinal params

b = np.reshape(a, newshape=(2,5) , order='C')

print(b)

""" >>> array([[0, 1, 2, 3, 4],
              [5, 6, 7, 8, 9]])
"""

b = np.reshape(a, newshape=(2,5) , order='F')

print(b)
""" >>> array([[0, 2, 4, 6, 8],
               [1, 3, 5, 7, 9]])
"""


```

you can filter numpy array with condition

``` python

a = np.arange(20)
b = a[a > 15]

print(b)

# >>> array([16, 17, 18, 19])

# or combine condition

a = np.arange(20)

b = a[(a < 10) & (a != 5)]

print(b)

# >>> array([0, 1, 2, 3, 4, 6, 7, 8, 9])

# for non zero element you can use np.nonzero()

a = np.arange(5)

b = np.nonzero(a)

print(b)

# >>> (array([1, 2, 3, 4], dtype=int64),)

```


## Numpy basic operations

### Arithmetic

This section covers addition, subtraction, multiplication and division 


``` python

a = np.arange(5,dtype=int)
b = np.ones(5 , dtype=int)

print(a + b)

# >>> array([1, 2, 3, 4, 5])

print(a - b)

# >>> array([-1,  0,  1,  2,  3])

c = b + 1

print(c)

# >>> array([2, 2, 2, 2, 2])

print(a * c)

# >>> array([0, 2, 4, 6, 8])

print(a / c)

# >>> array([0. , 0.5, 1. , 1.5, 2. ])

```

### Other operation 

This section covers sum, min, max, mean (avg) and more

``` python

#  sum() works for 1D arrays, 2D arrays, and arrays in higher dimensions.

a = np.arange(10)

print(a.sum())

# >>> 45

a = np.array([[1 ,15,4,8],[2, 8 , 9 ,2]])

print(a.sum())

# >>> 49

# you can specified axis

print(a.sum(axis=0))

# >>> array([ 3, 23, 13, 10])

print(a.sum(axis=1))

# >>> array([28, 21])

# for min use min()

print(a.min())

# >>> 1

# for max use max()

print(a.max())

# >>> 15

# for avg or mean use mean()

print(a.mean())

# >>> 6.125

# use prod() to get the result of multiplying the elements together

print(a.prod())

# >>> 138240

# use std() to get the standard deviation

print(a.std())

# >>> 4.456385867493972


```

To get unique items and counts


``` python

# use np.unique() to get unique items

a = np.array([11 ,2,15,2,10,4,47,45,44,8,44])

print(np.unique(a))

# >>> array([ 2,  4,  8, 10, 11, 15, 44, 45, 47])

# for indexes of unique items

unique_values , unique_indexes  = np.unique(a , return_index=True)

print(unique_indexes)

# >>> array([1, 5, 9, 4, 0, 2, 8, 7, 6], dtype=int64)

# to get counts of unique indexes 

unique_values , unique_count = np.unique(a, return_counts = True)

print(unique_values)

# >>> array([ 2,  4,  8, 10, 11, 15, 44, 45, 47])
print(unique_count)

# >>> array([2, 1, 1, 1, 1, 1, 2, 1, 1], dtype=int64)

``` 

For more information enter to help

``` python
# enter to numpy help
help(np)

# or one of np method

help(np.max)

```
