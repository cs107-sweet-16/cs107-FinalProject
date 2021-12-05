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

def mul(a,b):
    aval, ader = a 
    bval, bder = b 
    val = aval * bval
    der = dict()
    for k in ader:
        der[k] = ader[k] * bval
    for k in bder:
        if k in der:
            der[k] += bder[k] * aval
        else:
            der[k] = bder[k] * aval
    return val, der

def sub(a,b):
    aval, ader = a 
    bval, bder = b 
    val = aval - bval
    der = dict()
    for k in ader:
        der[k] = ader[k]
    for k in bder:
        if k in der:
            der[k] -= bder[k]
        else:
            der[k] = -bder[k]
    return val, der

def truediv(a,b):
    aval, ader = a 
    bval, bder = b 
    val = aval / bval
    der = dict()
    for k in ader:
        der[k] = ader[k] / bval
    for k in bder:
        if k in der:
            der[k] += - aval / bval / bval * bder[k]
        else:
            der[k] = - aval / bval / bval * bder[k]
    return val, der

def power(a, b):
    aval, ader = a 
    bval, bder = b 
    val = aval ** bval
    der = dict()
    for k in ader:
        der[k] =  ader[k]*bval*aval**(bval-1)
    for k in bder:
        if k in der:
            der[k] += val * np.log(aval) * bder[k]
        else:
            der[k] = val * np.log(aval) * bder[k]
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
        return funcNode(np.sin, np.cos, None, a, None)
    elif isinstance(a, int) or isinstance(a, float):
        n = valNode()
        n._set_val(np.sin(a))
        return n
    else:
        raise TypeError
    
    
def cosine(a):
    aval, ader = a 
    val = np.cos(aval)
    der = dict()
    for k in ader:
        der[k] = -np.sin(aval)*ader[k]
    return val, der


def cos(a):
    if isinstance(a, Node):
        return funcNode(cosine, a, None)
    elif isinstance(a, int) or isinstance(a, float):
        n = valNode()
        n._set_val(np.cos(a))
        return n
    else:
        raise TypeError


def tangent(a):
    aval, ader = a 
    val = np.tan(aval)
    der = dict()
    for k in ader:
        der[k] = ader[k]/(np.cos(aval))**2
    return val, der


def tan(a):
    if isinstance(a, Node):
        return funcNode(tangent, a, None)
    elif isinstance(a, int) or isinstance(a, float):
        n = valNode()
        n._set_val(np.tan(a))
        return n
    else:
        raise TypeError


def exponential(a):
    aval, ader = a 
    val = np.exp(aval)
    der = dict()
    for k in ader:
        der[k] = ader[k]*np.exp(aval)
    return val, der


def exp(a):
    if isinstance(a, Node):
        return funcNode(exponential, a, None)
    elif isinstance(a, int) or isinstance(a, float):
        n = valNode()
        n._set_val(np.exp(a))
        return n
    else:
        raise TypeError

def logarithm(a):
    aval, ader = a 
    val = np.log(aval)
    der = dict()
    for k in ader:
        der[k] = ader[k]/aval
    return val, der


def log(a):
    if isinstance(a, Node):
        return funcNode(logarithm, a, None)
    elif isinstance(a, int) or isinstance(a, float):
        n = valNode()
        n._set_val(np.log(a))
        return n
    else:
        raise TypeError
    

class Node:
    def __init__(self):
        # self.der = dict()
        pass
        
    

    
    def __add__(self, other):
        if isinstance(other, Node):
            r = other
        elif isinstance(other, int) or isinstance(other, float):
            r = valNode()
            r._set_val(other)
        else:
            raise TypeError
        return funcNode(np.add,lambda x,y: 1, lambda x,y: 1, self, r)
    
    def __radd__(self, other):
        return self + other
    
    
    def __mul__(self, other):
        if isinstance(other, Node):
            r = other
        elif isinstance(other, int) or isinstance(other, float):
            r = valNode()
            r._set_val(other)
        else:
            raise TypeError
        return funcNode(np.multiply,lambda x,y: y, lambda x,y: x, self, r)
    
    def __rmul__(self, other):
        return self * other
        

    def __neg__(self):
        return self * (-1)


    def __sub__(self, other):
        if isinstance(other, Node):
            r = other
        elif isinstance(other, int) or isinstance(other, float):
            r = valNode()
            r._set_val(other)
        else:
            raise TypeError
        return funcNode(sub,self,r)
    
    def __rsub__(self, other):
        return -self + other
    
    def __truediv__(self, other):
        if isinstance(other, Node):
            r = other
        elif isinstance(other, int) or isinstance(other, float):
            r = valNode()
            r._set_val(other)
        else:
            raise TypeError
        return funcNode(truediv,self,r)
    
    def __rtruediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            l = valNode()
            l._set_val(other)
        else:
            raise TypeError
        return funcNode(truediv,l,self)               

    def __pow__(self, other):
        if isinstance(other, Node):
            r = other
        elif isinstance(other, int) or isinstance(other, float):
            r = valNode()
            r._set_val(other)
        else:
            raise TypeError
        return funcNode(power, self, other)
    
    def __rpow__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            l = valNode()
            l._set_val(other)
        else:
            raise TypeError
        return funcNode(power, other, self)
        
    def __pos__(self):
        return self
    
    # logarithm
    # hyperbolic
    # square root
        
        
class valNode(Node):
    def __init__(self, name = None):
        super().__init__()
        self.der = 0
        self.name = name
        # if name != None:
        #     self.der[name]=1
    
    def _set_val(self, val):
        self.val = val
    def __str__(self):
        return str(self.name) if self.name != None else str(self.val)
    
    def forward(self):
        return self.val, {self.name: 1}
    
    def forward_pass(self):
        return self.val
        
    def reverse(self, partial, adjoint):
        if self.name != None:
            self.der += partial*adjoint


class funcNode(Node):
    def __init__(self, func, leftdf, rightdf, left, right):
        super().__init__()
        self.func = func
        self.leftdf = leftdf
        self.rightdf = rightdf
        self.val = None
        self.left = left
        self.right = right
    
    def __str__(self):
        return f'{self.func.__name__}' + \
               '\n|\n|-(L)->' + '\n|      '.join(str(self.left ).split('\n')) + \
               '\n|\n|-(R)->' + '\n|      '.join(str(self.right).split('\n'))
    
    def forward(self):
        if self.right != None:
            aval, ader = self.left.forward()
            bval, bder = self.right.forward()
            val = self.func(aval, bval)
            der = dict()
            for k in ader:
                der[k] = ader[k] * self.leftdf(aval, bval)
            for k in bder:
                if k in der:
                    der[k] += bder[k] * self.rightdf(aval, bval)
                else:
                    der[k] = bder[k] * self.rightdf(aval, bval)
            return val, der
        else:
            aval, ader = self.left.forward()
            val = self.func(aval)
            der = dict()
            for k in ader:
                der[k] = ader[k] * self.leftdf(aval)
            return val, der            
        
    def forward_pass(self):
        if self.right != None:
            self.val = self.func(self.left.forward_pass(), self.right.forward_pass())
        else:
            self.val = self.func(self.left.forward_pass())
        return self.val
    
    def reverse(self, partial, adjoint):
        if self.right != None:
            lder = self.leftdf(self.left.val, self.right.val)
            rder = self.rightdf(self.left.val, self.right.val)
            self.left.reverse(lder, partial*adjoint)
            self.right.reverse(rder, partial*adjoint)
        else:
            lder = self.leftdf(self.left.val)
            self.left.reverse(lder, partial*adjoint)     
            
            
if __name__=='__main__':
    a = valNode('a')
    b = valNode('b')
    f = sin(a * b + b)
    a._set_val(2)
    b._set_val(5)
    print(f.forward())
    f.forward_pass()
    print(f.val)
    f.reverse(1,1)
    print(a.der, b.der)
    
 