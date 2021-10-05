# Module 4 Object Oriented Programming in python

## What is OOP in python

Object-oriented programming (OOP) is a method of structuring a program by bundling related properties and behaviors into individual objects.

## create class

### syntax

Like function definitions begin with the def keyword in Python, class definitions begin with a class keyword.

``` python

class ThisIsMyNewClass:

    ''' this is a description of my new class'''

    name = 'aymen'

    def whoami(self):
        return 'jemi'

a = ThisIsMyNewClass()

a.name # result 'aymen' 

a.whoami() # result 'jemi'

```

You may have noticed the self parameter in function definition inside the class but we called the method simply as a.whoami() without any arguments. It still worked.

This is because, whenever an object calls its method, the object itself is passed as the first argument. So, a.whoami() translates into ThisIsMyNewClass.whoami(a).

For these reasons, the first argument of the function in class must be the object itself. This is conventionally called self. It can be named otherwise but we highly recommend to follow the convention.

### constructor

Class functions that begin with double underscore __ are called special functions as they have special meaning.

Of one particular interest is the __init__() function. This special function gets called whenever a new object of that class is instantiated.

This type of function is also called constructors in Object Oriented Programming (OOP). We normally use it to initialize all the variables.

``` python

class Student():

    def __init__(self, name='None', age=24, score=1.0):
        self.name = name
        self.age = age
        self.score = score

    def show(self):
        print('name is :', self.name)
        print('age is :', self.age)
        print('score is :', self.score)

# if we don't use params in constructor


student = Student()
student.show()

"""
name is : None
age is : 24
score is : 1.0
"""

# if we use params in constructor

student = Student('aymen',25,10.2)
student.show()

"""
name is : aymen
age is : 25
score is : 10.2
"""

```

## Deleting Attributes and Objects

``` python

student = Student('aymen',25,10.2)
student.show()

"""
name is : aymen
age is : 25
score is : 10.2
"""

del student.show
student.show()

# AttributeError: show



del student
student
# NameError: name 'student' is not defined

```


## Inheritance

### sample

``` python

class Person():
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def show(self):
        print('name is :', self.name)
        print('age is :', self.age)


class Student(Person):
    def __init__(self, name, age, score):
        Person.__init__(self, name, age)
        self.score = score

    def show(self):
        Person.show(self)
        print('score is :', self.score)


student = Student('aymen', 25, 10.2)

student.show()

```


### multiple

``` python

class Person():
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def show(self):
        print('name is :', self.name)
        print('age is :', self.age)


class Matieres():

    def __init__(self, title='none', cof=0.2, regime=''):
        self.title = title
        self.cof = cof
        self.regime = regime

    def show(self):
        print('Matiere title :', self.title)
        print('Matiere cof :', self.cof)
        print('Matiere regime :', self.regime)


class Student(Person, Matieres):
    def __init__(self, name, age, score, title, cof, regime):
        Person.__init__(self, name, age)
        Matieres.__init__(self, title, cof, regime)

        self.score = score

    def show(self):
        Person.show(self)
        Matieres.show(self)
        print('score is :', self.score)


student = Student('aymen', 25, 10.2, 'BA', 0.5, 'mixte')

student.show()

```