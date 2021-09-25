# Module 1 Introduction to Python and Python env (anaconda)

## General introduction

Python is an interpreted high-level general-purpose programming language. Its design philosophy emphasizes code readability with its use of significant indentation. Its language constructs as well as its object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects [source](https://en.wikipedia.org/wiki/Python_(programming_language))

## Anaconda

Anaconda is a distribution of the Python and R programming languages for scientific computing, that aims to simplify package management and deployment. The distribution includes data-science packages suitable for Windows, Linux, and macOS [source](https://en.wikipedia.org/wiki/Anaconda_(Python_distribution))

### install 

1) Go to the Anaconda [website](https://www.anaconda.com/products/individual).

2) Download Anaconda Installer (Win, Mac, Linux).

3) Follow the installer steps.


### details

run anaconda, add package with conda, try spyder IDE.

## Jupyter

Jupyter Notebook (formerly IPython Notebooks) is a web-based interactive computational environment for creating notebook documents.


A Jupyter Notebook document is a browser-based REPL (read eval print loop) containing an ordered list of input/output cells which can contain code, text (using Markdown), mathematics, plots and rich media.


Underneath the interface, a notebook is a JSON document, following a versioned schema, usually ending with the ".ipynb" extension. [source](https://en.wikipedia.org/wiki/Project_Jupyter)


## Visual Code

Visual Studio Code is a code editor made by Microsoft for Windows, Linux and macOS.[9] Features include support for debugging, syntax highlighting, intelligent code completion, snippets, code refactoring, and embedded Git. Users can change the theme, keyboard shortcuts, preferences, and install extensions that add additional functionality. [source](https://en.wikipedia.org/wiki/Visual_Studio_Code)


## Code and Variables

### Comments

``` python
# this is a line comment

""" 
This is
a Multi Line Comments .
"""
```

### Declaration

``` python
a = 2323
# or

a , b = "something" , 112.04
# or 

variable = Object()

```

### Sample Variable types

Numbers

``` python

#int 
a = 9

#float
a = 9.0

#complex
a = 1+2j

#Complex numbers are written in the form, x + yj, where x is the real part and y is the imaginary part.

#long not exist in python 3.x version sys.maxsize contains the maximum size in bytes a Python int can be

```

Strings

``` python

#string
a = "9"

#or
a = '9.0'

#or
a = "something"
# note cote or double cote is the same in python

```

Boolean

``` python

stop = False
start = True

# [T]rue and [F]alse the chars t and f are upercase.

```

### Input Output

input 

``` python

a = input('enter numeric value')

name = input('enter your name:')

```

output

``` python

a = input('enter numeric value')
# enter 5
print(a)
#result is >>> 5

#enter aymen jemi
name = input('enter your name:')
print(name)
#result is >>> aymen jemi
```


### Function
#### What’s a Function?
Every programming language lets you create blocks of code that, when called, perform tasks. [source](https://kidscodecs.com/programming-functions/)

``` python

#sample example of python function without params

def hello_word():
    name = input('enter your name:')
    print('Hello ', name)

#call hello_word function

hello_word()
# enter Word
# result is >>> Hello Word



# function with params

def hello_word_params(name):
    print('Hello ', name)

#call hello_word_params



hello_word_params('Word')

# result is >>> Hello Word




# function with init params

def my_func(name='Aymen',age=22):
    print('My name is :', name)
    print("I'am ",age, "years old")


# if you call without params
my_func()

"""
result is 
>>> My name is Aymen
>>> I'am 22 years old
"""
# if you enter params
my_func(name='Med')

"""
result is 
>>> My name is Med
>>> I'am 22 years old
"""

my_func(age=23,name='Med')
# or 
my_func('Med',23)

"""
result is 
>>> My name is Med
>>> I'am 23 years old
"""

"""
Note : if you enter the params names the orders dosen't matter
"""
```

#### inner function
Inner functions, also known as nested functions, are functions that you define inside other functions. In Python, this kind of function has direct access to variables and names defined in the enclosing function. [source](https://realpython.com/inner-functions-what-are-they-good-for/)


``` python

def my_function(name, age):
    def inner_print(intro, var):
        print(intro, var)
    
    inner_print("My name is ", name)

    inner_print("I'am ", age)

# call my_function

my_function('Med', 23)

"""
result is 
>>> My name is Med
>>> I'am 23
"""

# call inner_print function
inner_print("My name is ",'Med')
# result NameError: name 'inner_print' is not defined

```



### Usefull buildin Function


``` python

# float() Returns a floating number

float("2.7")
#result >>> 2.7
float(58)
#result >>> 58.0
float("445f55f8.")
# result >>> ValueError: could not convert string to float: '445f55f8.'


# int() Returns an integer number

int("55")
#result >>> 55

int("2.3")
#result >>> ValueError: invalid literal for int() with base 10: '2.3'

int("five")
#result >>> ValueError: invalid literal for int() with base 10: 'five'


# str() Returns a string object

str(55)
#result >>> '55'

str(188.2)
#result >>> '188.2'

a = 8
b = 18
str(a+b)

#result >>> '26'


# type() Returns the type of an object


a = 58
type(a)
#result >>> <class 'int'>
a = "name"
type(a)
#result >>> <class 'str'>
a = True
type(a)
#result >>> <class 'bool'>


# ord() Convert an integer representing the Unicode of the specified character

ord('h')
# result >>> 104

ord('aymen')
# result >>> TypeError: ord() expected a character, but string of length 5 found


# chr() Returns a character from the specified Unicode code.

chr(104)
# result >>> 'h'

# upper() Returns a string where all characters are in upper case.

'hello'.upper()
# >>> 'HELLO'

# lower() Returns a string where all characters are in lower case.

'Help Me'.lower()
# >>> 'help me'

```

## Exercises

1) write a python program to predict the next char.

if you enter a the program return b

2) write a python program to return the type of variable.

if you enter 'hello' the program return <class 'str'>

3) write a python program to convert string to number and string to float.

if you enter '1.2' the program return 1.2 and if you enter '2' the program return 2

4) write a python program to predict the next char with inner function.

if you enter a the program return b



