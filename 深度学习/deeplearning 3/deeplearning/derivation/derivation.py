
class Exp:
    def derivate(self):
        pass

    def __add__(self, other):
        return Add(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __truediv__(self, other):
        return Div(self, other)


class Constant(Exp):
    def __init__(self, value):
        self.value = value

    def derivate(self):
        return 0

    def __repr__(self):
        return str(self.value)


class Add(Exp):
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2

    def derivate(self):
        return Add(self.arg1.derivate(), self.arg2.derivate())

    def __repr__(self):
        return '(%s + %s)' % (self.arg1, self.arg2)


class Sub(Exp):
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2

    def derivate(self):
        return Sub(self.arg1.derivate(), self.arg2.derivate())

    def __repr__(self):
        return '(%s - %s)' % (self.arg1, self.arg2)


class X(Exp):
    def __init__(self):
        pass

    def derivate(self):
        return 1.0

    def __repr__(self):
        return 'X'


class Mul(Exp):
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2

    def derivate(self):
        op1 = Mul(self.arg1.derivate(), self.arg2)
        op2 = Mul(self.arg1, self.arg2.derivate())
        return Add(op1, op2)

    def __repr__(self):
        return '(%s * %s)' % (self.arg1, self.arg2)


class Div(Exp):
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2

    def derivate(self):
        op1 = Mul(self.arg1.derivate(), self.arg2)
        op2 = Mul(self.arg1, self.arg2.derivate())
        return Div(Sub(op1, op2), Mul(self.arg2, self.arg2))

    def __repr__(self):
        return '(%s / %s)' % (self.arg1, self.arg2)


if __name__ == '__main__':
    c1 = Constant(1234)
    c2 = Constant(5678)
    e = Add(c1, c2)
    # e = c1 - c2
    x = X()
    # e = Add(e, x)
    # e = Mul(c1, x)
    # e = Add(Mul(x, x), Mul(Constant(3), x))
    # e = Div(Constant(1.0), x)
    e = (c1+c2)/x
    print(e, e.derivate())
