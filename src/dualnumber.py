class Dualnumber:
    '''
        Implements the DualNumber class.
    '''

    def __init__(self, a):
        import numpy as np
        from dualnumber import Dualnumber
        self.val = a
        self.der = 1

    def set_dual(self, dual):
        self.der = dual

    def __add__(self, other):
        try:
            addition = Dualnumber(self.val + other.val)
            addition.der = self.der + other.der
            return addition
        except AttributeError as e:
            other = Dualnumber(other)
            addition = Dualnumber(self.val + other.val)
            addition.set_dual(self.der)
            return addition

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        try:
            multi = Dualnumber(self.val * other.val)
            multi.der = self.val * other.der + self.der * other.val
            return multi
        except AttributeError as e:
            multi = Dualnumber(self.val * other)
            multi.set_dual(self.der * other)
            return multi

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        try:
            subtraction = Dualnumber(self.val - other.val)
            subtraction.der = self.der - other.der
            return subtraction
        except AttributeError as e:
            other = Dualnumber(other)
            subtraction = Dualnumber(self.val - other.val)
            subtraction.set_dual(self.der)
            return subtraction

    def __rsub__(self, other):
        return self - other

    def __truediv__(self, other):
        try:
            div = Dualnumber(self.val/other.val)
            div.der = (self.der*other.val - self.val*other.der)/(other.val**2)
            return div
        except AttributeError as e:
            div = Dualnumber(self.val / other)
            div.set_dual(self.der / other)
            return div

    def __rdiv__(self, other):
        return self/other

    def __pow__(self, other):
        try:
            # dual nuumber raised to dual number
            pow = Dualnumber(self.val**other.val)
            pow.der = (other.val*(self.val**(other.val-1))*self.der) + (np.log(self.val))*(self.val**(other.val))*other.der
            return pow
        except AttributeError as e:
            # dual number raised to real number d^r
            pow = Dualnumber(self.val**other)
            pow.der = (other*self.val**(other-1))*self.der
            return pow
        # add in real number raised to dual number?

    ## check reverse power
    def __rpow__(self, other):
        pass



if __name__ == '__main__':
    # test passing 2 dual numbers to add
    x = Dualnumber(1)
    y = Dualnumber(2)
    z = x + y
    assert z.val == 3
    assert z.der == 2

    # test passing 1 dual and 1 non dual number to add
    x = Dualnumber(4)
    y = 5
    z = x + y
    assert z.val == 9
    assert z.der == 1

    # test passing 2 dual numbers to subtract
    x = Dualnumber(1)
    y = Dualnumber(2)
    z = x - y
    assert z.val == -1
    assert z.der == 0

    # test passing 1 dual and 1 non dual number to subtract
    x = Dualnumber(2)
    y = 5
    z = x - y
    assert z.val == -3
    assert z.der == 1

    # test passing 2 dual numbers through power:
    x = Dualnumber(2)
    y = Dualnumber(2)
    z = x**y
    assert z.val == 4
    assert np.isclose(z.der, 6.772588722239782)

    # test passing 1 dual number and 1 int through power:
    x = Dualnumber(2)
    y = 3
    z = x**y
    assert z.val == 8
    assert z.der == 12

    # test passing 2 dual numbers through division:
    x = Dualnumber(4)
    y = Dualnumber(2)
    z = x/y
    assert z.val == 2
    assert z.der == -0.5

    # test passing 1 dual number and 1 int through division:
    x = Dualnumber(4)
    y = 2
    z = x/y
    assert z.val == 2
    assert z.der == 0.5
