# %%
a = 2323
# or
a, b = "something", 112.04

# %%
a = input('enter numeric value')
print(a)
name = input('enter your name:')
print(name)
# %%

# sample example of python function without params


def hello_word():
    name = input('enter your name:')
    print('Hello ', name)

# call hello_word function


hello_word()
# %%


# function with params

def hello_word_params(name):
    print('Hello ', name)

# call hello_word_params


hello_word_params('Word')

# %%


# function with init params

def my_func(name='Aymen', age=22):
    print('My name is :', name)
    print("I'am ", age, "years old")


# if you call without params
my_func()

# %%
# if you enter params
my_func(name='Med')
# %%


my_func(age=23, name='Med')

# %%
# or
my_func('Med', 23)


# Note : if you enter the params names the orders dosen't matter
# %%

def my_function(name, age):
    def inner_print(intro, var):
        print(intro, var)

    inner_print("My name is ", name)

    inner_print("I'am ", age)

# call my_function


my_function('Med', 23)

# %%

# call inner_print function
inner_print("My name is ", 'Med')

# %%
float("2.7")
# %%
float(58)

# %%
float("445f55f8.")
# %%

int("55")

# %%
int("2.3")
# %%
int("five")
# %%
str(55)

# %%
str(188.2)

# %%
a = 8
b = 18
str(a+b)
# %%
a = 58
type(a)
# %%
a = "name"
type(a)
# %%
a = True
type(a)

# %%
ord('h')
# %%
ord('aymen')
# %%
chr(104)
# %%
'hello'.upper()
# %%
'Help Me'.lower()


# %%
"""
--------------------------------------
                Exercises
--------------------------------------
"""

# ex 4


def next_char(char):

    def inner_char(number_char):
        print(chr(number_char + 1))

    number_char = ord(char)
    inner_char(number_char)


next_char('1')
