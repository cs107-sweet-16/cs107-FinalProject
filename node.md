# node module


### class node.Node()
Bases: `object`

Node class that implements the following native python functions: addition, subtraction, multipliation,
division, unary operators, and power. Depending on the implemented funtions, Node will check if the inputs
are of type integer, float, or Node. If they are of tyep Node, then the value and it’s derivative is computed
and stored in a dictionary. If the values are of type int or float, then the values are first convered to Node
class by calling valNode.


#### val()
Current value of the variable.


* **Type**

    int, float



#### der()
Derivative of variable.


* **Type**

    int, float



#### add()
Addition implementation for Node class.


#### sub()
Subtraction implementation for Node class.


#### mul()
Multiplication implementation for Node class.


#### div()
Division implementation for Node class.


#### pow()
Power implementation for Node class.


#### pos()
Positive unary operator for Node class.


#### neg()
Negative unary operator for Node class.


### node.cos(a)
Implements cosine to contribute to a computational graph. Converts integers and floats to
Node’s, and inserts derivatives into the computational graph.


* **Parameters**

    **a** (*int**, **float**, **Node*) – Value to be calculated through cosine.



* **Returns**

    Node with values and derivatives along with a corresponding computational graph.



* **Raises**

    **TypeError if value is not of type int****, ****float****, or ****Node.** – 



### node.cosh(a)
Implements hyperbolic cosine to contribute to a computational graph. Converts integers and floats to
Node’s, and inserts derivatives into the computational graph.


* **Parameters**

    **a** (*int**, **float**, **Node*) – Value to be calculated through hyperbolic cosine.



* **Returns**

    Node with values and derivatives along with a corresponding computational graph.



* **Raises**

    **TypeError if value is not of type int****, ****float****, or ****Node.** – 



### node.exp(a)
Implements exponential function to calculate values, derivatives, and to contribute to a computational graph. Converts integers and floats to
Node’s, and inserts derivatives into the computational graph.


* **Parameters**

    **a** (*int**, **float**, **Node*) – Value to be raised to the power of e.



* **Returns**

    Node with values and derivatives along with a corresponding computational graph.



* **Raises**

    **TypeError if value is not of type int****, ****float****, or ****Node.** – 



### class node.funcNode(func, leftdf, rightdf, left, right)
Bases: `node.Node`


#### forward()

#### forward_pass()

#### reverse(partial, adjoint)

### node.ln(a)
Implements the natural logarithm to calculate values, derivative, and to contribute to a computational graph. Converts integers and floats to
Node’s, and inserts derivatives into the computational graph.


* **Parameters**

    **a** (*int**, **float**, **Node*) – Value to be passed through natural logarithm.



* **Returns**

    Node with values and derivatives along with a corresponding computational graph.



### node.log(a)
Alias for ln(a).


* **Parameters**

    **a** (*int**, **float**, **Node*) – Value to be passed through natural logarithm.



* **Returns**

    Node with values and derivatives along with a corresponding computational graph.



### node.log_ab(a, b)

### node.logistic(a)

### node.sin(a)
Implements sine to contribute to a computational graph. Converts integers and floats to
Node’s, and inserts derivatives into the computational graph.


* **Parameters**

    **a** (*int**, **float**, **Node*) – Value to be calculated through sine.



* **Returns**

    Node with values and derivatives along with a corresponding computational graph.



* **Raises**

    **TypeError if value is not of type int****, ****float****, or ****Node.** – 



### node.sinh(a)
Implements hyperbolic sine to contribute to a computational graph. Converts integers and floats to
Node’s, and inserts derivatives into the computational graph.


* **Parameters**

    **a** (*int**, **float**, **Node*) – Value to be calculated through hyperbolic sine.



* **Returns**

    Node with values and derivatives along with a corresponding computational graph.



* **Raises**

    **TypeError if value is not of type int****, ****float****, or ****Node.** – 



### node.sqrt(a)
Implements square-root to contribute to a computational graph. Converts integers and floats to
Node’s, and inserts derivatives into the computational graph.


* **Parameters**

    **a** (*int**, **float**, **Node*) – Value to be square-rooted.



* **Returns**

    Node with values and derivatives along with a corresponding computational graph.



* **Raises**

    **TypeError if value is not of type int****, ****float****, or ****Node.** – 



### node.tan(a)
Implements tangent to contribute to a computational graph. Converts integers and floats to
Node’s, and inserts derivatives into the computational graph.


* **Parameters**

    **a** (*int**, **float**, **Node*) – Value to be calculated through tan.



* **Returns**

    Node with values and derivatives along with a corresponding computational graph.



* **Raises**

    **TypeError if value is not of type int****, ****float****, or ****Node.** – 



### node.tanh(a)
Implements hyperbolic tangent to contribute to a computational graph. Converts integers and floats to
Node’s, and inserts derivatives into the computational graph.


* **Parameters**

    **a** (*int**, **float**, **Node*) – Value to be calculated through hyperbolic tangent.



* **Returns**

    Node with values and derivatives along with a corresponding computational graph.



* **Raises**

    **TypeError if value is not of type int****, ****float****, or ****Node.** – 



### class node.valNode(name=None)
Bases: `node.Node`

Sets variables as valNode class, which allows the users to auto-differentiate functions that use them. valNode class
overrides native python functions and unary operators, in order to construct a sequential computational graph. Sets
the initial derivative (0) for each assigned variable.


#### forward()
Executes a simple forward pass for a single node.


* **Returns**

    Current value and empty derivative if number. Otherwise returns the value of the specified valNode and
    empty dictionary.



#### forward_pass()

#### reverse(partial, adjoint)
