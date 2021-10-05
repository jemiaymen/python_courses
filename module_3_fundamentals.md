# Module 3 Python Fundamentals

## Condition

sample condition

``` python

if (1 == 1):
    print('yes')

```

multiple condition


``` python

if (1 != 1):
    print('no')
elif (2 != 2):
    print('no 2')
elif (3 == 3):
    print('yes 3')
else:
    print('noting')


if (1 == 1) or ('b' == 'b'):
    print('is the same')


```

nested if or inner if

``` python

if ('aymen' != 'jemi'):
    print('someting')
    if (type('aymen') == type('jemi')):
        print('same types')
    else:
        print('not same types')
```


## Loops

### while

With the while loop we can execute a set of statements as long as a condition is true.

``` python
i = 1
while i < 6:
    print(i)
    i += 1

```

### range and for loop
The range() function returns a sequence of numbers, starting from 0 by default, and increments by 1 (by default), and stops before a specified number. [ range(start, stop, step) ]

``` python


for n in range(10):
    print(n)


x = range(3, 6)
for n in x:
    print(n)


x = range(3, 20, 2)
for n in x:
  print(n)

```

## Continue and break

``` python

for x in range(20):
    if (x == 10):
        break
    print(x)

i = 0
while (True):
    if (i % 2 == 0):
        continue
    print(i)

    if (i == 20):
        break
    i+= 1

for x in range(20):

    if (x == 15):
        continue

    if ((x + 1) % 2 == 0):
        print(x)
    
```


## Try

The try block lets you test a block of code for errors.

The except block lets you handle the error.

The finally block lets you execute code, regardless of the result of the try and except blocks.

``` python


try:
    # x = 1 / 0
    x = int('2.1')
except ZeroDivisionError:
    print("Division error")
except Exception:
    print("Something else went wrong")


try:
    x = 1 / 0
    # x = int('2.1')
except ZeroDivisionError:
    print("Division error")
except Exception:
    print("Something else went wrong")

```


## Python Modules
``` python
#diff between import and from module import

import numpy as np
import pandas as pd

from math import pow

print( pow(2,5))


```

## PIP and CONDA tools
``` python
#in cli pip install something

#in cli conda install 

```

## Python program structure
``` python
#what is module and how to create module and import this module
# see modules/m_1.py and modules/m_2.py


```

## Exercises