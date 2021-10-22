# module student

def show(name,age ,score ):
    print('name is :', name)
    print('age is :' , age)
    print('score is :', score)

def calcule_moyenne(mat):
    moy = 0
    for x in mat:
        moy += x
    return moy / len(mat)