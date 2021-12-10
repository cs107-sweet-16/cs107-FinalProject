<!-- autodiff documentation master file, created by
sphinx-quickstart on Tue Nov  9 21:28:25 2021.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive. -->
# Welcome to autodiff’s documentation!

# API documentation:

* node module
* Dualnumber module
* Operatorsfunc modeul


# Introduction

Differentiation is an important operation in scientific research broadly
applied in non-linear optimization problems and numerical solution of
differential equations. Recently, the development of machine learning
techniques including artificial neural networks also require large amount of
gradient computation. For complex functions whose analytical derivative
expressions are difficult to obtain, one traditional option to calculate its
derivative is finite-difference method, but may have issues such as
floating point errors. Automatic differentiation (AD) refers to a general way
of computing the derivatives of functions expressed as computation graphs, with
which the evaluation of derivatives could reach machine precision. Our autodiff
package provides tools to calculate the derivative of any function with an
analytic expression based on the AD technique.

# Background

AD mainly makes use of two properties of closed-form functions:
1. Any closed-form function is a combination (sum, product, quotient or compound) of
elementary functions, and the analytical expressions of the derivatives of
elementary functions are already-known.
2. Derivative rules including sum rule, product rule, quotient rule and most importantly, the chain rule.
- sum rule: (f+g)’ = f’+g’
- product rule: (f\*g)’ = f’g+g’f\*g
- quotient rule: (f/g)’ = (f’g-g’f)/(g×g)
- chain rule: h = f(g(•)), then h’ = f’(g(•))× g’(•)

To evaluate the value of a complicated function expression, we need to start with
independent variables and evaluate a series of intermediate results and finally
reach the final result. The idea of AD is to use a computational graph to
represent the function where each node represents an intermediate variable and
only elementary operations are carried out between intermediate variables. Because we
utilize the operation rules of derivatives mentioned above, the derivative at any
node in the computation graph can only rely on its parent node(s). As a result, we
obtain the final derivative result by computing derivatives and repeatedly utilizing
derivative rules following the flow of the computation graph.

Here we give a brief example to illustrate how we calculate derivatives 
based on compution graph. If we want to evaluate the derivative with respect 
to x1 at x1=pi/2 for function f=sin(cos(x1)), here is our computation graph

![comp graph](fig1.png)

Then we follow the graph and evaluate step by step:
|step| trace | Elementary function |Current value | Elementary function derivative| $\nabla_x$ value|
|--|--|--|--|-- |--|
|1|v0|x|pi/2|$\dot x$|1|
|2|v1|cos(v0)|0|$-sin(v_0)\dot v_0$|0|
|3|f|sin(v1)|0|$cos(v_1)\dot v_1$|0|

Note that when we calculate "elementary function derivative", we made use of the chain rule. 

# How to use

## Installation

### Option 1:

We utilized PyPi to make our autodifferentiation package publicly available. Users may install autodiff (?? is this the name) through the following command line:

```
pip install autodiff
```

Within a python terminal, you can then import the package:

```
import autodiff
```

Alternatively, to pull commmon functions directly you may use the following import statement:

```
from autodiff.node import sin, cos, tan, exp, ln, log, sinh, cosh, tanh, sqrt, logistic, log_ab, valNode
```

### Option 2:

In adition to pip installation, you may clone directly from the repository:

```
git clone https://github.com/cs107-sweet-16/cs107-FinalProject.git
```

After successful cloning of the repository, follow by installing the package requirements from the ‘requirements.txt’ file.

```
cd cs107-FinalProject
pip3 install -r requirements.txt
```

Finally, finish by importing dualnumber and it’s operators

```
import numpy as np
from node import sin, cos, tan, exp, log, valNode
```
## Interaction

Autodiff executes auto-differntiation by utilizting the concept of a computational graph. This graph is constructed during the execution of a forward pass. Simultaneously, the final value, and the respective gradient of each variable are also computed. Forward mode may be passed on some simple user-defined function, e.g. $$ f = 3x + 5^y $$. The computational graph is constructed via Python's native operator precedence and an ordered dictionaries. As each step is executed per Python's operator precedence, the respective values and derivatives are stored within the dictionary. The code for this may be found under the directory `autodiff/`

### Forward mode

The general workflow for this project consists of the following:
1. Determining some function of interest
2. Initializing each unique variable 
3. Defining the function
4. Setting the variable values
5. Executing forward mode
6. Executing reverse pass

It is important to note that may of these steps must be defined in order. Particularly the following two considerations:
1. The variables must be set as nodes prior to function definition. This ensures that as the function is execute the values will be simultaneously calculated and the computational graph will be constructed.
2. Forward mode is executed prior to reverse mode. For reverse() to be completed, the gradients at each node and the final gradient must be computed beforehand, through forward(). Otherwise, reverse() fails.

```
  # testing cos(ab/c) + c*log(a), a = 4, b = -1, c = 10

  # initialize the nodes as variables"
  a = valNode('a')
  b = valNode('b')
  c = valNode('c')

  # define the function:
  f = cos((a * b) / c) + c * log(a)
  a.set_val(4)
  b.set_val(-1)
  c.set_val(10)

  # execute forward:
  f_val, f_grad = f.forward()


 # print the values of interest:
 print("value of function f: ", f_val)
 print("gradient with respect to a: ", f_grad[a])
 print("gradient with respect to b: ", f_grad[b])
 print("gradient with respect to c: ", f_grad[c])

```

### Forward and Reverse Mode

```


## EXAMPLE 2 with reverse mode:
 # declare a as node
  a = valNode('a')
 # set a's value
  a._set_val(1.731)

 # set function f: a^2 + 3^a + (5/1) + a/2
  f = a**2 + 3**a + 5/a + a/2

 # execute forward pass
  f.forward_pass()
  f_val, f_grad = f.reverse(1, 1)

 # print the values of interest:
 print("value of function f: ", f_val)
 print("gradient at of a (1,1): ", f_der)
```

### Simple Usage

In addition to our node.py file, which contains code for forward and reverse passes on a given function, we also have more rudimentary code that constructs dual numbers and conducts a forward pass. These files are: `Dualnumber.py` and `Operatorsfunc.py`. Both of which can be found along within the `autodiff/` diretory.

This code is similar to the previous code, except there is no reverse pass available, and the forward pass executes automatically when a function is passed. Examples below:

```py
# set dual number 'x'
# first value (np.pi) is the real part at which we want to calculate the derivative and value of the function
# second value '1' is the dual part. In this case it is set to one, and will default to 1, but there is an option to change it. 
x = Dualnumber(np.pi, der=1)

# Next implement your function, by substituting in the defined dual number:
f = (tan(x)) - 2 ** x * exp(x)

# The forward pass has now been executed and we can check the value of the function and it's derivative:
# value of the function: 
print(f.val) # should equal approx -204.2160993
# value of the derivative:
print(f.der) # should equal approx -344.76791290283
```

## Software Organization

### Directory Structure
Our code will live in a separate folder called `src`. We will have a separate
directory, `test`, for our tests. Finally, while not necessarily code, our
documentation will live in a directory named `docs`, which will go into
technical detail how each user-facing function is used, and any examples if
appropriate.

`
CS107-FinalProject/
  autodiff/
    Dualnumber.py
    Operatorsfunc.py
    node.py
  docs/
    source/
      ...
    documentation.md
    milestone1.md
    milestone2.md
    milestone2_progress.md
  test/
    __init__.py
    test_node.py
    test_dual.py
    test_node_errs.py
    test_operatorsfunc.py
`

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

Our test suite will live in a separate directory called `tests`. We will use
TravisCI to run continuous integration, and make sure we do not accidentally
introduce regressions in our code when we change or add functionality. CodeCov
will be used to ensure that any new code that we write is properly tested and
accounted for.

### Documentation
We will use Sphinx as our documentation generator. To build sphinx, you can run the command inside
the `docs/` directory:
```
sphinx-build source/ build/
```

The generated HTML files are viewable as `cs107-FinalProject/docs/build/index.html`.
    
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
algebraic operations, along with their reverse counterparts (not shown). 

```py
class Dualnumber:
    def __init__(self, a, der=1):
        self.val = a
        self.der = der

    def set_dual(self, dual):
        self.der = dual

    def __add__(self, other):

    def __mul__(self, other):

    def __sub__(self, other):

    def __truediv__(self, other):

    def __pow__(self, other):

    def __neg__(self, other):
    
    def __pos__(self, other):
    
    def __eq__(self, other):
```

### Primitives
We will have a module that is dedicated to housing elementary operations. 
We will maintain an updated list of these operations:
1. `sin`
2. `cos`
3. `pow`
4. `exp`
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

## Future Features
Our future features include the following:
* Fine-tuning our forward pass to include a computational graph
* Adding additional functions like the inverse of sinusoidal functions, and other operations
* Utilizing the computational graph to execute a backward pass
* Creating more advanced documentation through Sphinx 
* Releasing the package on PyPi for installation through pip
* Expand our testing suite to accomodate additional features

We expect the most difficult part of our future features to be constructing the computational graph. We expect to accomplish this through ordered dictionaries, where we will add lines of code to each defined function, so when they are called upon we can access the operations and numbers in order. We will then use the operations and numbers to contruct the nodes and edges of a computational graph, which will subsequently be used fto perform the backward pass. 



