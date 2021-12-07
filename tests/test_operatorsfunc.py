import pytest
import sys
import os
import numpy as np

# sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")
from autodiff.dualnumber import Dualnumber
from autodiff.operatorsfunc import sin, cos, tan, exp, log


def test_sample():
    assert True == True


def test_sin_float():
    # sin test passing in a float
    x = np.pi
    sin_x = sin(x)
    assert np.isclose(sin_x.val, 0)
    assert sin_x.der == 0


def test_sin_dualnumber():
    # sin test passing in a dualnumber
    x = Dualnumber(np.pi)
    x.set_dual(np.pi / 2)
    sin_x = sin(x)
    assert np.isclose(sin_x.val, 0)
    assert sin_x.der == - np.pi / 2


def test_cos_float():
    # cos test passing in a float
    x = np.pi
    cos_x = cos(x)
    assert np.isclose(cos_x.val, -1)
    assert cos_x.der == 0


def test_cos_dualnumber():
    # cos test passing in a dualnumber
    x = Dualnumber(np.pi)
    x.set_dual(np.pi / 2)
    cos_x = cos(x)
    assert np.isclose(cos_x.val, -1)
    assert np.isclose(cos_x.der, 0)


def test_tan_float():
    # tan test passing in a float
    x = np.pi / 4
    tan_x = tan(x)
    assert np.isclose(tan_x.val, 1)
    assert tan_x.der == 0


def test_tan_dualnumber():
    # tan test passing in a dualnumber
    x = Dualnumber(np.pi / 4)
    x.set_dual(np.pi)
    tan_x = tan(x)
    assert np.isclose(tan_x.val, 1)
    assert np.isclose(tan_x.der, 2 * np.pi)


def test_exp_float():
    # exp test passing in a float
    x = 2
    exp_x = exp(x)
    assert np.isclose(np.exp(2), exp_x.val)
    assert exp_x.der == 0


def test_exp_dualnumber():
    # exp test passing in a dualnumber
    x = Dualnumber(2)
    x.set_dual(1)
    exp_x = exp(x)
    assert np.isclose(np.exp(2), exp_x.val)
    assert np.isclose(np.exp(2), exp_x.der)


def test_log_float():
    # log test  for float
    x = np.exp(1)
    log_x = log(x)
    assert np.isclose(1, log_x.val)
    assert np.isclose(0, log_x.der)


def test_log_dualnumber():
    # log test for Dual number
    x = Dualnumber(np.exp(1), 2)
    log_x = log(x)
    assert np.isclose(1, log_x.val)
    assert np.isclose(np.exp(-1) * 2, log_x.der)


def test_complex_function_1():
    # complex function 1
    x = Dualnumber(2, der=1)
    f = sin(x ** 2 + x) - log(x)
    assert np.isclose(-0.9725626, f.val)
    assert np.isclose(4.300851433251, f.der)


def test_complex_function_2():
    # complex function 2
    x = Dualnumber(np.pi, der=1)
    f = (tan(x)) - 2 ** x * exp(x)
    assert np.isclose(-204.2160993, f.val)
    assert np.isclose(-344.76791290283, f.der)


def test_complex_function_3():
    # complex function 3
    x = Dualnumber(np.pi / 4, der=np.pi / 4)
    f = tan(x) + x
    print(f.der)
    assert np.isclose(f.val, np.pi / 4 + 1)
    assert np.isclose(f.der, 3 * x.der)


def test_complex_function_4():
    # complex function 4
    x = Dualnumber(2, der=1)
    f = exp(x) + log(x) * sin(x) ** 2
    assert np.isclose(f.val, np.exp(x.val) + np.log(x.val) * np.sin(x.val) ** 2)
    assert np.isclose(f.der, np.exp(x.val) + np.sin(x.val) * np.sin(x.val) / x.val + np.log(x.val) * np.sin(2 * x.val))
