
class A:
    def __init__(self):
        self.y = 1234567

    def glossory(self):
        return 30

    def f(self):
        return 'abcde'

    def gg(self):
        return self.glossory() * self.y


class B(A):
    def __init__(self, p):
        print('p=' + p)
        super(B, self).__init__()

    def h(self):
        return 100

    def glossory(self):
        return -30

    # def g(self):
    #     print('in B.g()')
    #     return 3000


if __name__ == '__main__':
    b = B('123321')
    print(b.gg())
