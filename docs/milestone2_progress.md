# Milestone 2

## Progress Update

### Core autodiff functionality
This past week we have focused on writing some basic code for the 
project and discussing future directions. The former includes 
coding our `dualNumber` class. This class will override native 
python functions in order to apply them to dual numbers (e.g. 
addition, division). These function can be found in 
'dualnumber.py'. To address other,  non-native elementary functions 
(e.g. sinusoidal), we have begun to write independent class 
statements using numpy. These can be found in 'operators.py'. For 
each operation, we have included test functions in the respective 
files. These will eventually be moved out into their own testing suite. 

Next week, we plan on implementing a forward class, through which 
we will simultaneously structure a computational class. We plan on
 achieving this through an ordered dictionary. We will provide 
test cases to test the functionality of our automatic differentiation 
function, e.g., foot-finding algorithm. We also plan to make our package 
able to manage forward-mode automatic differentiation with multiple 
independent variables, and will provide a test case to apply our package
in gradient descent algorithm. As an additional component to our project,
 we hope to also implement reverse pass 
by utilizing the computational graph created in the forward pass. 

### Testing infrastructure and documentation
We are starting to set up all the necessary integration tooling 
to ensure a smooth development process. By milestone 2, we will have
* TravisCI for continuous integration (running tests and ensure no regressions are introduced)
* Codecov for code coverage (every line of code written is properly tested)
* Sphinx documentation (autodoc will generate documentation for us based on docstrings in the code)

### Group member assignments
Bhawesh and Anna have been assigned the task to complete writing 
basic operations and elementary functions for milestone 2. We 
will also be writing the test cases for those operators and functions.

Jingxing has been assigned the task to write test cases implementing root
finding algorithm and gradient descent algorithm using our package to 
calculate derivatives. Jingxing will also provide the `requirements.txt` file 
for users to install necessary dependencies.

Albert is responsible for guiding the team on high-level designs, and enforcing good and
consistent software practices. A good engineering leader is able to clear auxiliary roadblocks
for their team, so that the team can focus on as much core functionality as possible. To that end,
Albert is responsible for the following items:
* Providing a overview (in pseudocode) of how to implement forward pass for milestone2
* Setting up TravisCI, Codecov, Sphinx
* Setting up pytest
* Providing examples of how to write clean, modular, testable code

