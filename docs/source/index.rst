.. autodiff documentation master file, created by
   sphinx-quickstart on Tue Nov  9 21:28:25 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to autodiff's documentation!
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   dual
   operators
   node


Introduction
============

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

Background
==========
AD mainly makes use of two properties of closed-form functions: 
1. Any closed-form function is a combination (sum, product, quotient or compound) of
elementary functions, and the analytical expressions of the derivatives of
elementary functions are already-known.
2. Derivative rules including sum rule, product rule, quotient rule and most importantly, the chain rule.  
- sum rule: (f+g)' = f'+g'
- product rule: (f*g)' = f'g+g'f*g
- quotient rule: (f/g)' = (f'g-g'f)/(g×g)
- chain rule: h = f(g(•)), then h' = f'(g(•))× g'(•)

To evaluate the value of a complicated function expression, we need to start with
independent variables and evaluate a series of intermediate results and finally
reach the final result. The idea of AD is to use a computational graph to
represent the function where each node represents an intermediate variable and
only elementary operations are carried out between intermediate variables. Because we
utilize the operation rules of derivatives mentioned above, the derivative at any 
node in the computation graph can only rely on its parent node(s). As a result, we 
obtain the final derivative result by computing derivatives and repeatedly utilizing 
derivative rules following the flow of the computation graph. 


How to use
==========
To use the current version of our autodifferntiation python package, you must first 'git clone' on your terminal.

.. code-block:: console

  git clone https://github.com/cs107-sweet-16/cs107-FinalProject.git

After successful cloning of the repository, follow by installing the package requirements from the 'requirements.txt' file.

.. code-block:: console

  cd cs107-FinalProject
  pip3 install -r requirements.txt


Finally, finish by importing dualnumber and it's operators

.. code-block:: Python

  import numpy as np
  from src.dualnumber import Dualnumber
  from src.operatorsfunc import sin, cos, tan, exp, log

Forward mode
------------

.. code-block:: Python

  ## EXAMPLE 1:
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


  ## EXAMPLE 2:
  # Here is a second example, executed similarly to the first except with the initial dual part not equal to 1

  # set the dual number, with dual = pi/4
  x = Dualnumber(np.pi / 4, der=np.pi / 4)

  # forward pass:
  f = tan(x) + x

  # check value:
  print(f.val)

  # The true derivative of a function for a dual number where the dual part is not equal to one, must be divided by the initial dual part.
  # i.e. f.der/x.der = true derivative at x.val
  # Check derivative:
  print(f.der) # should equal approx: 3 * x.der or 3 * pi/4, where 3 is the actual derivative at x.val=np.pi/4

  # Check value
  print(f.val) # should be equal to np.pi / 4 + 1



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

