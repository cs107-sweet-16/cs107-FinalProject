# Milestone 1

## Introduction
Differentiation is an important operation in scientific research broadly
applied in non-linear optimization problems and numerical solution of
differential equations. Recently, the development of machine learning
techniques including artificial neural networks also require large amount of
gradient computation. For complex functions whose analytical derivative
expressions are difficult to obtain, one traditional option to calculate its
derivative is finite-difference method, but may have issues such as
floating point errors. Automatic differentiation (AD) refers to a general way
of computing the derivatives of functions expressed as computation graphs, with
which the evaluation of derivatives could reach machine precision. Our `autodiff`
package provides tools to calculate the derivative of any function with an
analytic expression based on the AD technique. 

## Background

AD mainly makes use of two properties of closed-form functions: 
1. Any closed-form function is a combination (sum, product, quotient or compound) of
elementary functions, and the analytical expressions of the derivatives of
elementary functions are already-known.
2. Derivative rules including sum rule, product rule, quotient rule and most importantly, the chain rule.  
- sum rule: (f+g)' = f'+g'
- product rule: (f*g)' = f'g+g'f*g
- quotient rule: (f/g)' = (f'g-g'f)/(g×g)
- chain rule: h = f(g(•)), then h' = f'(g(•))× g'(•)

To evaluate the value of a complex function expression, we need to start with
independent variables and evaluate a series of intermediate results and finally
reach the final result. The idea of AD is to use a computational graph to
represent the function where each node represents an intermediate variable and
only elementary operations are carried out between intermediate variables. Because we
utilize the operation rules of derivatives mentioned above, the derivative at any 
node in the computation graph can only rely on its parent node(s). As a result, we 
obtain the final derivative result by computing derivatives and repeatedly utilizing 
derivative rules following the flow of the computation graph. 

Here we give an brief example to illustrate how we calculate 

%%%Therefore, we follow the computational graph to evaluate the function value at
any certain node. 
To evaluate the derivative of a function at a certain point,
we make use of chain rule: the derivative of the composition h = f(g(•))
of two differentiable functions f and g is h' = f'(g(•))×
g'(•) which ensures that the derivative with respect to independent
variables of a node in the computational graph is the multiplication of the
derivative of the previous node with respect to the independent variables and
the derivative of the current node with respect to the previous node. As a
result, the derivative of the function can also be calculated automatically
based on the flow of the computational graph, and no asymptotic assumptions are
needed in the evaluation process.%%%

## How to Use 
```py
from autodiff import gradient
from autodiff.primitives import symbol
x = symbol()
f = x ** 2 
grad_f = gradient(f) # this computes the gradient of f, not yet evaluated at a particular value
grad_f(2) # this would evaluate the gradient of f at the value, so would return 4

```
## Software Organization

### Directory Structure
Our code will live in a separate folder called `src`. We will have a separate
directory, `test`, for our tests. Finally, while not necessarily code, our
documentation will live in a directory named `docs`, which will go into
technical detail how each user-facing function is used, and any examples if
appropriate.

### Modules

We will have one package, named `autodiff`. Within this module there may be a
few modules, with the most core functionality in a module named `grad`.  This
module will provide us with the basic functionality of being able to
differentiate on a defined set of elementary operations.  To help define the
scope of our project, we will determine ahead of time which elementary
functions can be used to integrate with our library. In order to accomplish
this, we will create a second module, `primitives`, which houses some of the
elementary operations we define in the implementation section, as well as the 
symbol class.

### Test Suite

Our test suite will live in a separate directory called `test`. We will use
TravisCI to run continuous integration, and make sure we do not accidentally
introduce regressions in our code when we change or add functionality. CodeCov
will be used to ensure that any new code that we write is properly tested and
accounted for.
    
### Distribution 
We will use PyPI to distribute our package.   
We will use a library called `twine` to help us package and distribute our
package over PyPI. We will follow this helpful tutorial
https://packaging.python.org/tutorials/packaging-projects/, provides a detailed
guide on how to setup the software structure. Using a framework may be helpful,
but in an effort to reduce software overhead and maximize learning for each
step, we will elect to forgo a framework.

## Implementation 
We will have a few core data structures:
* Computational graph
* Custom classes for instantiating dual
numbers and performing forward and backward passes.

### Computational Graph
The computational graph will consist of node and edge data structures. 
The graph is bidirectional since we plan to implement both the backward pass and forward pass 
implementations.
The nodes will store data in tuples.

### Symbol Class 
The symbol class is a user-facing data structure which represents 
an abstraction of a variable. It will use dual numbers under the hood
that run on the computational graph.

### Dual Number Class 
We will begin by constructing a dualNumber() class that
deals with the properties of dual numbers. It will take a real value and dual
value as the initial input. The dualNumber class will have the following
algebraic operations.

```py
class dualNumber():
  def __init__(self, real, dual):
    self.real = real
    self.dual = dual
  def __sum__(self, other):
  def __subtract__(self, other):
  def __mul__(self, other):
  def __div__(self, other):
```

### Primitives
We will have a module that is dedicated to housing elementary operations. 
We will maintain an updated list of these operations:
1. `sin`
2. `cos`
3. `pow`
4.  `exp`
5. `log`

### Forward Pass 
The exact details of the forward pass algorithm are currently abstracted,
but we desire to implement a forward mode function which can take in a function,
and utilize dual numbers and computational graph
to compute the resulting value and respective gradients.

### Reverse Pass
The exact details of the reverse pass algorithm are currently abstracted,
but we desire to implement a reverse mode function which will implement
the backpropagation step, utilizing the saved results from our forward pass.

## License
We will use the MIT License for our project because we are intending
to publish the project on github as open-source. MIT License allows for
commercial and private use, while protecting us against liability and warranty.
Furthermore, MIT License is BSD compatible, which is the license used by Numpy,
so we do not need to call any other license.
