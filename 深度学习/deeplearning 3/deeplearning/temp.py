#  Annotation

# AOP     Aspect-Oriented Programming


def m(f):
    print('in m')

    def ff(x, y):
        print('call ff')
        if x >= 10 or y >= 10:
            raise Exception('Please do not add numbers that is equal to or great than 10.')
        return f(x, y)
    return ff


@m
def my_func(a, b):
    return a+b


@m
def x(p, q):
    return 10


if __name__ == '__main__':
    # print(my_func(3, 4))
    # print(my_func(5, 6))
    # print(my_func(15, 6))
    print(x(4, 5))
