import numpy as np 




def sin(a):
    if isinstance(a, Node):
        return funcNode(np.sin, np.cos, None, a, None)
    elif isinstance(a, int) or isinstance(a, float):
        n = valNode()
        n.set_val(np.sin(a))
        return n
    else:
        raise TypeError

def cos(a):
    if isinstance(a, Node):
        return funcNode(np.cos, lambda x: -np.sin(x), None, a, None)
    elif isinstance(a, int) or isinstance(a, float):
        n = valNode()
        n.set_val(np.cos(a))
        return n
    else:
        raise TypeError

def tan(a):
    if isinstance(a, Node):
        return funcNode(np.tan, lambda x: 1/(np.cos(x))**2, None, a, None)
    elif isinstance(a, int) or isinstance(a, float):
        n = valNode()
        n.set_val(np.tan(a))
        return n
    else:
        raise TypeError

def sinh(a):
    if isinstance(a, Node):
        return funcNode(np.sinh, np.cosh, None, a, None)
    elif isinstance(a, int) or isinstance(a, float):
        n = valNode()
        n.set_val(np.sinh(a))
        return n
    else:
        raise TypeError
    
def cosh(a):
    if isinstance(a, Node):
        return funcNode(np.cosh, np.sinh, None, a, None)
    elif isinstance(a, int) or isinstance(a, float):
        n = valNode()
        n.set_val(np.cosh(a))
        return n
    else:
        raise TypeError

def tanh(a):
    if isinstance(a, Node):
        return funcNode(np.tanh, lambda x: 1-np.tanh(x)**2, None, a, None)
    elif isinstance(a, int) or isinstance(a, float):
        n = valNode()
        n.set_val(np.tanh(a))
        return n
    else:
        raise TypeError

def logistic(a):
    pass

def sqrt(a):
    if isinstance(a, Node):
        return funcNode(np.sqrt, lambda x: 0.5/np.sqrt(x), None, a, None)
    elif isinstance(a, int) or isinstance(a, float):
        n = valNode()
        n.set_val(np.sqrt(a))
        return n
    else:
        raise TypeError


def exp(a):
    if isinstance(a, Node):
        return funcNode(np.exp, np.exp, None, a, None)
    elif isinstance(a, int) or isinstance(a, float):
        n = valNode()
        n.set_val(np.exp(a))
        return n
    else:
        raise TypeError

def ln(a):
    if isinstance(a, Node):
        return funcNode(np.log, lambda x: 1/x, None, a, None)
    elif isinstance(a, int) or isinstance(a, float):
        n = valNode()
        n.set_val(np.log(a))
        return n
    else:
        raise TypeError

def log(a,b):
    pass
    

class Node:
    def __init__(self):
        # self.der = dict()
        pass
    
    def __add__(self, other):
        if isinstance(other, Node):
            r = other
        elif isinstance(other, int) or isinstance(other, float):
            r = valNode()
            r.set_val(other)
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
            r.set_val(other)
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
            r.set_val(other)
        else:
            raise TypeError
        return funcNode(np.subtract,lambda x,y: 1, lambda x,y: -1, self, r)
    
    def __rsub__(self, other):
        return -self + other
    
    def __truediv__(self, other):
        if isinstance(other, Node):
            r = other
        elif isinstance(other, int) or isinstance(other, float):
            r = valNode()
            r.set_val(other)
        else:
            raise TypeError
        return funcNode(np.divide, lambda x,y: 1/y, lambda x,y: -x/y/y, self, r)
    
    def __rtruediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            l = valNode()
            l.set_val(other)
        else:
            raise TypeError
        return funcNode(np.divide, lambda x,y: 1/y, lambda x,y: -x/y/y, l, self)               

    def __pow__(self, other):
        if isinstance(other, Node):
            r = other
        elif isinstance(other, int) or isinstance(other, float):
            r = valNode()
            r.set_val(other)
        else:
            raise TypeError("unsupported input type(s) for the operand")
        return funcNode(np.power, lambda x,y: y*np.power(x,y-1), lambda x,y: np.power(x,y)*np.log(x), self, r)
    
    def __rpow__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            l = valNode()
            l.set_val(other)
        else:
            raise TypeError
        return funcNode(np.power, lambda x,y: y*np.power(x,y-1), lambda x,y: np.power(x,y)*np.log(x), l, self)
        
    def __pos__(self):
        return self

        
        
class valNode(Node):
    def __init__(self, name = None):
        super().__init__()
        self.der = 0
        self.name = name
        # if name != None:
        #     self.der[name]=1
    
    def set_val(self, val):
        self.val = val
    def __str__(self):
        return str(self.name) if self.name != None else str(self.val)
    
    def forward(self):
        if self.name != None:
            return self.val, {self.name: 1}
        else:
            return self.val, {}
    
    def forward_pass(self):
        self.der = 0
        return self.val
        
    def reverse_pass(self, partial, adjoint):
        if self.name == None:
            return dict()
        self.der += partial*adjoint
        return {self.name: self}

    def _reset_der(self):
        self.der = 0

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
    
    def reverse_pass(self, partial, adjoint):
        if self.right != None: 
            lder = self.leftdf(self.left.val, self.right.val)
            rder = self.rightdf(self.left.val, self.right.val)
            lvars = self.left.reverse_pass(lder, partial*adjoint)
            rvars = self.right.reverse_pass(rder, partial*adjoint)
            variables = dict()
            for v in rvars:
                if v not in lvars:
                    lvars[v] = rvars[v]
        else:
            lder = self.leftdf(self.left.val)
            lvars = self.left.reverse_pass(lder, partial*adjoint) 
            
        return lvars
        # return self.val
    
    def reverse(self):
        self.forward_pass()
        variables = self.reverse_pass(1,1)
        der = { v: node.der for v, node in variables.items() }
        return self.val, der
    
    
class vector:
    def __init__(self, *args):
        self.size = len(args)
        self.elements = args
    
    def set_val(self, array):
        if len(array) != self.size:
            raise ValueError(f"Input size has a mismatch with the vector size ({self.size})")
        for node, val in zip(self.elements, array):
            # print(val)
            node.set_val(val)

    def __getitem__(self, key):
        return self.elements[key]
        
    def __iter__(self, key):
        return iter(self.elements)
    
    def grad(self, var):
        '''
            auto-diff for vector functions
            currently we only use reverse mode to calculate gradient for vector functions
            input: the variable or vector variables var, which is used to
             define the function and for which we need to calculate the derivatives
            return: Jacobian matrix
        '''
        der_array = []
        var._reset_der()
        for f in self.elements:
            f.forward_pass()
            f.reverse_pass(1,1)
            der_array.append(var.der)
            var._reset_der()
        return der_array
    
    def evaluate(self):
        '''
            evaluation for vector functions
        '''
        return [ f.forward_pass() for f in self.elements ]        
    
    @property
    def der(self):
        return [ node.der for node in self.elements ]
    
    def _reset_der(self):
        '''
           this function only works if it is a variable vector
        '''
        for node in self.elements:
            node.der = 0
    
def variables(name, size = None):
    if size == None or size == 1:
        return valNode(name)
    elems = [ valNode(name+'___'+str(i)) for i in range(size) ]
    return vector(*elems)

    

            
if __name__=='__main__':

    
    # v = vector([1,2,3])

    # example to define scalar function with multiple input variables

    x = variables('x')
    y = variables('y')
    f = sin(ln(x))+tan(x*x+y*x+x*x*x*y)
    x.set_val(2)
    y.set_val(3)
    print(f.forward())
    print(f.reverse())
    
    # example to define vector function with vectorized input variables
    
    v = variables('v', 2)
    v.set_val([2,3])    
    print(v.evaluate())
    print(v.grad(v))
    def func(v):
        '''
            f takes a size=3 vector and output a size=2 vector
        '''
        f1 = sin(ln(v[0]))+tan(v[0]**2+v[0]*v[1]+v[0]**3*v[1])
        f2 = v[0]*v[1]+v[1]**2+sqrt(v[0])
        return vector(f1,f2)
    f = func(v)
    print(f.evaluate())
    print(f.grad(v))
        
        
    
    # vector function can also take a scalar input variable
    def func1(x):
        f1 = x**2
        f2 = x**3
        return vector(f1, f2)
    
    f = func1(x)
    print(f.evaluate())
    print(f.grad(x))
        
    '''x.der = 0
    y.der = 0
    f = log(sin(exp(x)+y)) + exp(y) + x
    x._set_val(0)
    y._set_val(1)
    print(f.forward())
    f.forward_pass()
    f.reverse(1,1)
    print(f.val, x.der, y.der)
    
    x.der = 0
    y.der = 0
    f = log(sin(x+y)**2)+exp(x**2+y**2)
    print(f)
    x._set_val(1)
    y._set_val(2)
    print(f.forward())
    f.forward_pass()
    f.reverse(1,1)
    print(f.val, x.der, y.der)
    
    print("testing: sin(ab + b)")
    a = valNode('a')
    b = valNode('b')
    f = sin(a * b + b)
    a._set_val(2)
    b._set_val(5)
    # print(f)
    # print(f.forward())
    f_val, f_grad = f.forward()

    actual_f_val  = np.sin(a.val * b.val + b.val)
    actual_f_grad = {
        'a': b.val * np.cos((a.val + 1) * b.val), 
        'b': (a.val + 1) * np.cos((a.val + 1) * b.val)
    }

    assert np.isclose(f_val, actual_f_val)
    
    for var in f_grad:
        assert np.isclose(f_grad[var], actual_f_grad[var])

    f.forward_pass()
    # print(f.val, )
    f.reverse(1,1)
    reverse_grads = {
        'a': a.der,
        'b': b.der
    }
    for var in f_grad:
        assert np.isclose(f_grad[var], reverse_grads[var])
    
    print("testing: e^(a/c) + b")
    a = valNode('a')
    b = valNode('b')
    c = valNode('c')
    f = exp(a / c) + b
    a._set_val(np.pi/2)
    b._set_val(np.pi/3)
    c._set_val(np.pi)

    actual_f_val = np.exp(a.val / c.val) + b.val
    # print(np.exp(a.val / c.val), np.arccos(np.exp(a.val / c.val)))

    actual_f_grad = {
        'a': (np.exp(a.val / c.val)) / (c.val),
        'b': 1,
        'c': -(a.val * np.exp(a.val / c.val)) / (c.val ** 2),
    }

    f.forward_pass()
    # print(f.val, )
    f.reverse(1,1)
    reverse_grads = {
        'a': a.der,
        'b': b.der,
        'c': c.der,
    }

    for var in actual_f_grad.keys():
        assert np.isclose(actual_f_grad[var], reverse_grads[var])'''



 
