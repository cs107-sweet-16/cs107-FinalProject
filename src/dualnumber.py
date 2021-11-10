class Dualnumber:

    def __init__(self, a):
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
        pass

    def __div__(self, other):
        pass
