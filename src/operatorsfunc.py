from dualnumber import Dualnumber
import numpy as np
from typing import Union
import math


def sin(x: Union[float, Dualnumber]) -> Dualnumber:
    try:
        sin_x = Dualnumber(np.sin(x.val))
        sin_x.set_dual(np.cos(x.val) * x.der)
        return sin_x
    except AttributeError as e:
        sin_x = Dualnumber(np.sin(x))
        sin_x.set_dual(0)
        return sin_x


def cos(x: Union[float, Dualnumber]) -> Dualnumber:
    try:
        cos_x = Dualnumber(np.cos(x.val))
        cos_x.set_dual(-np.sin(x.val) * x.der)
        return cos_x
    except AttributeError as e:
        cos_x = Dualnumber(np.cos(x))
        cos_x.set_dual(0)
        return cos_x


def tan(x: Union[float, Dualnumber]) -> Dualnumber:
    tan_x = sin(x) / cos(x)
    return tan_x


def exp(x: Union[float, Dualnumber]) -> Dualnumber:
    # this is specifically euler's number?
    try:
        exp_x = Dualnumber(np.exp(x.val))
        exp_x.set_dual((x.val - 1) * np.exp(x.val) * x.der)
        return exp_x
    except AttributeError as e:
        exp_x = Dualnumber(np.exp(x))
        exp_x.set_dual(0)
        return exp_x


def log(x: Union[float, Dualnumber]) -> Dualnumber:
    try:
        log_x = Dualnumber(x.val)
        log_x.der = (1 / x.val) * x.der
        return log_x
    except AttributeError as e:
        log_x = Dualnumber(x)
        log_x.set_dual(0)
        return log_x


if __name__ == '__main__':
    # sin test passing in a float
    x = np.pi
    sin_x = sin(x)
    assert np.isclose(sin_x.val, 0)
    assert sin_x.der == 0

    # sin test passing in a dualnumber
    x = Dualnumber(np.pi)
    x.set_dual(np.pi / 2)
    sin_x = sin(x)
    assert np.isclose(sin_x.val, 0)
    assert sin_x.der == - np.pi / 2

    # cos test passing in a float
    x = np.pi
    cos_x = cos(x)
    assert np.isclose(cos_x.val, -1)
    assert cos_x.der == 0

    # cos test passing in a dualnumber
    x = Dualnumber(np.pi)
    x.set_dual(np.pi / 2)
    cos_x = cos(x)
    assert np.isclose(cos_x.val, -1)
    assert np.isclose(cos_x.der, 0)

    # tan test passing in a float
    x = np.pi / 4
    tan_x = tan(x)
    assert np.isclose(tan_x.val, 1)
    assert tan_x.der == 0

    # tan test passing in a dualnumber
    x = Dualnumber(np.pi / 4)
    x.set_dual(np.pi)
    tan_x = tan(x)
    assert np.isclose(tan_x.val, 1)
    assert np.isclose(tan_x.der, 2 * np.pi)

    # exp test passing in a float
    x = 1
    exp_x = exp(x)
    assert np.isclose(2.718281828459045, exp_x.val)
    assert exp_x.der == 0

    # exp test passing in a dualnumber
    x = Dualnumber(2)
    x.set_dual(1)
    exp_x = exp(x)
    assert np.isclose(2.718281828459045 ** 2, exp_x.val)
    assert np.isclose(exp_x.der, 2.718281828459045 ** 2)
