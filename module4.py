# %%
class ThisIsMyNewClass:

    ''' this is a description of my new class'''

    name = 'aymen'

    def whoami(self, name):
        return name


a = ThisIsMyNewClass()

a.name
# %%
a.whoami('jemi')
# %%


# class Student():

#     def __init__(self, name='None', age=24, score=1.0):
#         self.name = name
#         self.age = age
#         self.score = score

#     def show(self):
#         print('name is :', self.name)
#         print('age is :', self.age)
#         print('score is :', self.score)

# if we don't use params in constructor


student = Student()
student.show()

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


class Student(Person):
    # def __init__(self, name, age, score):
    #     Person.__init__(self, name, age)
    #     self.score = score

    def __init__(self, name, age, score):
        super().__init__(name, age)
        self.score = score

    def show(self):
        super().show()
        print('score is :', self.score)


student = Student('aymen', 25, 10.2)

student.show()

# %%
