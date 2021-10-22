# %%
class ThisIsMyNewClass():

    ''' this is a description of my new class'''

    name = 'aymen'

    def whoami(self, name):
        return name


# %%
a = ThisIsMyNewClass()
print(a.name)
print(a.whoami('jemi'))
# %%


class Student():

    def __init__(self):
        self.name = None
        self.age = None
        self.score = None

    def show(self):
        print('Name is :', self.name)
        print('Age is :', self.age)
        print('Score is :', self.score)


# %%
student = Student()
student.show()

# %%

# if we use params in constructor

student = Student('aymen', 25, 10.2)
student.show()
# %%

del student.show
student.show()
# %%
del student
student
# %%


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

# %%

class Salle():
    def __init__(self, salle_name=''):
        self.salle_name = salle_name

    def print(self):
        print('Salle est : ', self.salle_name)

class Course():
    pass

class Day():
    pass

class Module():
    pass

