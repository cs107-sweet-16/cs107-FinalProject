import pytest
import sys
import os
import numpy as np

# sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")
from autodiff.node import sin, cos, tan, exp, ln, log, sinh, cosh, tanh, sqrt, logistic, log_ab, valNode

def test_valnode_str():
    node = valNode('aasdfasdjfaks')
    assert str(node)  == 'aasdfasdjfaks'


def test_node_add_type_error():
    nodeA = valNode('aasdfasdjfaks')
    nodeB = 'b'
    with pytest.raises(TypeError):
        nodeA + nodeB

    with pytest.raises(TypeError):
        nodeB + nodeA

def test_node_pow_type_error():
    nodeA = valNode('aasdfasdjfaks')
    nodeB = 'b'
    with pytest.raises(TypeError):
        nodeA ** nodeB

    with pytest.raises(TypeError):
        nodeB ** nodeA

def test_node_sub_type_error():
    nodeA = valNode('aasdfasdjfaks')
    nodeB = 'b'
    with pytest.raises(TypeError):
        nodeA - nodeB

    with pytest.raises(TypeError):
        nodeB - nodeA

def test_node_mul_type_error():
    nodeA = valNode('aasdfasdjfaks')
    nodeB = 'b'
    with pytest.raises(TypeError):
        nodeA * nodeB

    with pytest.raises(TypeError):
        nodeB * nodeA


def test_node_div_type_error():
    nodeA = valNode('aasdfasdjfaks')
    nodeB = 'b'
    with pytest.raises(TypeError):
        nodeA / nodeB

    with pytest.raises(TypeError):
        nodeB / nodeA
