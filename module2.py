
# ex 1

def calculator(expression):
    if ('+' not in expression and '*' not in expression and '/' not in expression and '-' not in expression):
        print('Expression is wrong')
    elif ('+' in expression):
        result = expression.split('+')
        print(float(result[0]) + float(result[1]))
    elif ('-' in expression):
        result = expression.split('-')
        print(float(result[0]) - float(result[1]))
    elif ('*' in expression):
        result = expression.split('*')
        if(float(result[0]) == 0 or float(result[1]) == 0):
            print('0')
        else:
            print(float(result[0]) * float(result[1]))
    elif ('/' in expression):
        result = expression.split('/')
        if (float(result[0]) == 0):
            print('0')
        elif (float(result[1]) == 0):
            print('error devide by zeros')
        else:
            print(float(result[0]) / float(result[1]))


calculator("12/0")
