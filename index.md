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

Autodiff executes auto-differntiation by utilizting the concept of a computational graph. This graph is constructed during the execution of a forward pass. Simultaneously, the final value, and the respective gradient of each variable are also computed. Forward mode may be passed on some simple user-defined function, e.g. $f = 3x + 5^y$. The computational graph is constructed via Python's native operator precedence and an ordered dictionaries. As each step is executed per Python's operator precedence, the respective values and derivatives are stored within the dictionary. 

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




