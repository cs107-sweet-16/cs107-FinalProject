import numpy as np 


def add(a,b):
    aval, ader = a 
    bval, bder = b 
    val = aval + bval
    der = dict()
    for k in ader:
        der[k] = ader[k]
    for k in bder:
        if k in der:
            der[k] += bder[k]
        else:
            der[k] = bder[k]
    return val, der


def sine(a):
    aval, ader = a 
    val = np.sin(aval)
    der = dict()
    for k in ader:
        der[k] = np.cos(aval)*ader[k]
    return val, der


def sin(a):
    if isinstance(a, Node):
        return funcNode(sine, a, None)
    elif isinstance(a, int) or isinstance(a, float):
        n = valNode()
        n._set_val(np.sin(a))
        return n
    else:
        raise TypeError
    
    

class Node:
    def __init__(self):
        self.der = dict()
    

    
    def __add__(self, other):
        if isinstance(other, Node):
            r = other
        elif isinstance(other, int) or isinstance(other, float):
            r = valNode()
            r._set_val(other)
        else:
            raise TypeError
        return funcNode(add,self,r)
    

        
        
        
class valNode(Node):
    def __init__(self, name = None):
        super().__init__()
        self.name = name
        if name != None:
            self.der[name]=1
    
    def _set_val(self, val):
        self.val = val
    def __str__(self):
        return str(self.name) if self.name != None else str(self.val)
    
    def forward(self):
        return self.val, self.der


class funcNode(Node):
    def __init__(self, func, left, right):
        super().__init__()
        self.func = func
        self.left = left
        self.right = right
    
    def __str__(self):
        return f'{self.func.__name__}' + \
               '\n|\n|-(L)->' + '\n|      '.join(str(self.left ).split('\n')) + \
               '\n|\n|-(R)->' + '\n|      '.join(str(self.right).split('\n'))
    
    def forward(self):
        if self.right != None:
            return self.func(self.left.forward(), self.right.forward())
        else:
            return self.func(self.left.forward())
        


if __name__=='__main__':
    a = valNode('a')
    b = valNode('b')
    c = sin(a+1+b)
    print(c)
    a._set_val(2)
    b._set_val(3)
    print(c.forward())
    
 