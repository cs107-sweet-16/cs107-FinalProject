import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")
from node import sin, cos, tan, exp, log, sinh, cosh, tanh, sqrt, logistic, log_ab, valNode


def test_sin_multivar():
    print("testing: sin(ab + b) + c^a")
    a = valNode('a')
    b = valNode('b')
    c = valNode('c')
    f = sin(a * b + b) + c ** a
    a._set_val(2)
    b._set_val(5)
    c._set_val(3)

    actual_f_val = np.sin(a.val * b.val + b.val) + (c.val) ** a.val
    actual_f_grad = {
        'a': b.val * np.cos((a.val + 1) * b.val) + (c.val) ** (a.val) * np.log(c.val),
        'b': (a.val + 1) * np.cos((a.val + 1) * b.val),
        'c': (a.val) * (c.val) ** (a.val - 1)
    }
    f.forward_pass()
    # print(f.val, )
    f.reverse(1, 1)
    reverse_grads = {
        'a': a.der,
        'b': b.der,
        'c': c.der,
    }
    assert f.val == actual_f_val
    for var in actual_f_grad:
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_sin_multivar_2():
    # print("testing: sin(ab + b)")
    a = valNode('a')
    b = valNode('b')
    f = sin(a * b + b)
    a._set_val(2)
    b._set_val(5)
    # print(f)
    # print(f.forward())
    f_val, f_grad = f.forward()

    actual_f_val = np.sin(a.val * b.val + b.val)
    actual_f_grad = {
        'a': b.val * np.cos((a.val + 1) * b.val),
        'b': (a.val + 1) * np.cos((a.val + 1) * b.val)
    }

    # TODO this is for forward mode - move to different test file
    assert np.isclose(f_val, actual_f_val)

    for var in f_grad:
        assert np.isclose(f_grad[var], actual_f_grad[var])
    #

    f.forward_pass()
    # print(f.val, )
    f.reverse(1, 1)
    reverse_grads = {
        'a': a.der,
        'b': b.der
    }
    assert f.val == actual_f_val
    for var in f_grad:
        assert np.isclose(f_grad[var], reverse_grads[var])


def test_exp_multivar():
    a = valNode('a')
    b = valNode('b')
    c = valNode('c')
    f = exp(a / c) + b
    a._set_val(np.pi / 2)
    b._set_val(np.pi / 3)
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
    f.reverse(1, 1)
    reverse_grads = {
        'a': a.der,
        'b': b.der,
        'c': c.der,
    }
    assert f.val == actual_f_val
    for var in actual_f_grad.keys():
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_tan_multivar():
    a = valNode('a')
    b = valNode('b')
    a._set_val(1.5)
    b._set_val(2)

    f = tan(a) - exp(b)
    f.forward_pass()
    f.reverse(1, 1)
    reverse_grads = {
        'a': a.der,
        'b': b.der
    }

    actual_f_val = np.tan(a.val) - np.exp(b.val)
    actual_f_grad = {
        'a': (1 / np.cos(a.val)) ** 2,
        'b': -np.exp(b.val)
    }
    assert f.val == actual_f_val
    for var in actual_f_grad.keys():
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_log_multivar():
    a = valNode('a')
    b = valNode('b')
    c = valNode('c')
    d = valNode('d')
    a._set_val(1.5)
    b._set_val(2)
    c._set_val(np.pi / 4)
    d._set_val(3)

    f = d * log(a) + log(exp(b)) + log(cos(c))
    f.forward_pass()
    f.reverse(1, 1)
    reverse_grads = {
        'a': a.der,
        'b': b.der,
        'c': c.der,
        'd': d.der,
    }

    actual_f_val = d.val * np.log(a.val) + np.log(np.exp(b.val)) + np.log(np.cos(c.val))
    actual_f_grad = {
        'a': d.val * 1 / a.val,
        'b': 1,
        'c': -1 / np.cos(c.val) * np.sin(c.val),
        'd': np.log(a.val),
    }
    assert f.val == actual_f_val
    for var in actual_f_grad.keys():
        # print(actual_f_grad[var], reverse_grads[var])
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_tan_multivar2():
    a = valNode('a')
    b = valNode('b')
    a._set_val(2)
    b._set_val(np.pi)

    f = tan(exp(a / b))
    f.forward_pass()
    f.reverse(1, 1)
    reverse_grads = {
        'a': a.der,
        'b': b.der,
    }

    actual_f_val = np.tan(np.exp(a.val / b.val))
    actual_f_grad = {
        'a': np.exp(a.val / b.val) * 1 / (np.cos(np.exp(a.val / b.val)) ** 2) / (b.val),
        'b': -a.val * (np.exp(a.val / b.val)) * 1 / (np.cos(np.exp(a.val / b.val)) ** 2) / (b.val ** 2),
    }
    assert f.val == actual_f_val
    for var in actual_f_grad.keys():
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_pow_div_multivar():
    a = valNode('a')
    a._set_val(1.731)

    f = a ** 2 + 3 ** a + 5 / a + a / 2
    f.forward_pass()
    f.reverse(1, 1)
    reverse_grads = {
        'a': a.der,
    }
    actual_f_val = a.val ** 2 + 3 ** a.val + 5 / a.val + a.val / 2
    actual_f_grad = {
        'a': 2 * a.val + 3 ** a.val * np.log(3) - 5 / (a.val ** 2) + 1 / 2
    }
    assert f.val == actual_f_val
    for var in actual_f_grad.keys():
        # print(actual_f_grad[var], reverse_grads[var])
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_logistic():
    a = valNode('a')
    a._set_val(2)
    b = valNode('b')
    b._set_val(3)

    f = logistic(sin(a) + cos(b))
    f.forward_pass()
    f.reverse(1, 1)
    reverse_grads = {
        'a': a.der,
        'b': b.der
    }
    actual_f_val = 0.5 * (1 + np.tanh(0.5 * np.sin(a.val) + 0.5 * np.cos(b.val)))
    actual_f_grad = {
        'a': actual_f_val * (1 - actual_f_val) * np.cos(a.val),
        'b': -1 * actual_f_val * (1 - actual_f_val) * np.sin(b.val)
    }
    assert f.val == actual_f_val
    for var in actual_f_grad.keys():
        # print(actual_f_grad[var], reverse_grads[var])
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_log_ab_both_nodes():
    a = valNode('a')
    a._set_val(2)
    b = valNode('b')
    b._set_val(3)

    f = log_ab(a + b, a * b)
    f.forward_pass()
    f.reverse(1, 1)
    reverse_grads = {
        'a': a.der,
        'b': b.der
    }
    actual_f_val = (np.log(a.val + b.val)) / (np.log(a.val * b.val))
    actual_f_grad = {
        'a': (((np.log(a.val * b.val)) / (a.val + b.val)) - ((np.log(a.val + b.val)) / a.val)) / (
                    (np.log(a.val * b.val)) ** 2),
        'b': (((np.log(a.val * b.val)) / (a.val + b.val)) - ((np.log(a.val + b.val)) / b.val)) / (
                    (np.log(a.val * b.val)) ** 2)
    }
    assert f.val == actual_f_val
    for var in actual_f_grad.keys():
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_log_ab_base_int_arg_node():
    a = valNode('a')
    a._set_val(np.exp(2))

    f = sin(log_ab(a, 3))
    f.forward_pass()
    f.reverse(1, 1)
    reverse_grads = {
        'a': a.der,
    }
    actual_f_val = np.sin(np.log(a.val) / np.log(3))
    actual_f_grad = {
        'a': np.cos(np.log(a.val) / np.log(3)) * (1 / (a.val * np.log(3)))
    }
    assert f.val == actual_f_val
    for var in actual_f_grad.keys():
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_log_ab_base_Node_arg_int():
    a = valNode('a')
    a._set_val(np.exp(2))

    f = sin(log_ab(3, a))
    f.forward_pass()
    f.reverse(1, 1)
    reverse_grads = {
        'a': a.der
    }
    actual_f_val = np.sin(np.log(3) / np.log(a.val))
    actual_f_grad = {
        'a': (-np.cos(np.log(3) / np.log(a.val)) * np.log(3) * 1 / (a.val * np.log(a.val) * np.log(a.val)))
    }
    assert f.val == actual_f_val
    for var in actual_f_grad.keys():
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_log_ab_base_arg_int():
    a = valNode('a')
    a._set_val(np.exp(2))

    f = sin(log_ab(log_ab(np.exp(3), np.exp(1)), a))
    f.forward_pass()
    f.reverse(1, 1)
    reverse_grads = {
        'a': a.der
    }
    actual_f_val = np.sin(np.log(3) / np.log(a.val))
    actual_f_grad = {
        'a': (-np.cos(np.log(3) / np.log(a.val)) * np.log(3) * 1 / (a.val * np.log(a.val) * np.log(a.val)))
    }
    assert f.val == actual_f_val
    for var in actual_f_grad.keys():
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_sinh1():
    a = valNode('a')
    b = valNode('b')
    c = valNode('c')

    a._set_val(1)
    b._set_val(2)
    c._set_val(np.pi / 4)

    f = log(sinh(a)) + log(exp(b)) + log(cos(c))
    f.forward_pass()
    f.reverse(1, 1)
    reverse_grads = {
        'a': a.der,
        'b': b.der,
        'c': c.der,
    }

    actual_f_val = np.log(np.sinh(a.val)) + np.log(np.exp(b.val)) + np.log(np.cos(c.val))
    actual_f_grad = {
        'a': 1 / np.tanh(a.val),
        'b': 1,
        'c': -np.tan(c.val),
    }
    assert f.val == actual_f_val
    for var in actual_f_grad.keys():
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_sinh2():
    a = valNode('a')
    b = valNode('b')

    a._set_val(1)
    b._set_val(2)

    f = tan(sinh(a) * log(b))
    f.forward_pass()
    f.reverse(1, 1)
    reverse_grads = {
        'a': a.der,
        'b': b.der,
    }

    actual_f_val = np.tan(np.sinh(a.val) * np.log(b.val))
    actual_f_grad = {
        'a': np.cosh(a.val) * np.log(b.val) * ((1 / np.cos(np.sinh(a.val) * np.log(b.val))) ** 2),
        'b': np.sinh(a.val) * ((1 / np.cos(np.sinh(a.val) * np.log(b.val))) ** 2) / b.val,
    }
    assert f.val == actual_f_val
    for var in actual_f_grad.keys():
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_cosh1():
    a = valNode('a')
    b = valNode('b')
    c = valNode('c')

    a._set_val(1)
    b._set_val(2)
    c._set_val(np.pi / 4)

    f = log(cosh(a)) + log(exp(b)) + log(cos(c))
    f.forward_pass()
    f.reverse(1, 1)
    reverse_grads = {
        'a': a.der,
        'b': b.der,
        'c': c.der,
    }

    actual_f_val = np.log(np.cosh(a.val)) + np.log(np.exp(b.val)) + np.log(np.cos(c.val))
    actual_f_grad = {
        'a': np.tanh(a.val),
        'b': 1,
        'c': -np.tan(c.val),
    }
    assert f.val == actual_f_val
    for var in actual_f_grad.keys():
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_cosh2():
    a = valNode('a')
    b = valNode('b')

    a._set_val(1)
    b._set_val(2)

    f = tan(cosh(a) * log(b))
    f.forward_pass()
    f.reverse(1, 1)
    reverse_grads = {
        'a': a.der,
        'b': b.der,
    }

    actual_f_val = np.tan(np.cosh(a.val) * np.log(b.val))
    actual_f_grad = {
        'a': np.sinh(a.val) * np.log(b.val) * ((1 / np.cos(np.cosh(a.val) * np.log(b.val))) ** 2),
        'b': np.cosh(a.val) * ((1 / np.cos(np.cosh(a.val) * np.log(b.val))) ** 2) / b.val,
    }
    assert f.val == actual_f_val
    for var in actual_f_grad.keys():
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_tanh1():
    a = valNode('a')
    b = valNode('b')
    c = valNode('c')

    a._set_val(1)
    b._set_val(2)
    c._set_val(np.pi / 4)

    f = log(tanh(a)) + log(exp(b)) + log(cos(c))
    f.forward_pass()
    f.reverse(1, 1)
    reverse_grads = {
        'a': a.der,
        'b': b.der,
        'c': c.der,
    }

    actual_f_val = np.log(np.tanh(a.val)) + np.log(np.exp(b.val)) + np.log(np.cos(c.val))
    actual_f_grad = {
        'a': 1/(np.sinh(a.val)*np.cosh(a.val)),
        'b': 1,
        'c': -np.tan(c.val),
    }
    assert f.val == actual_f_val
    for var in actual_f_grad.keys():
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_tanh2():
    a = valNode('a')
    b = valNode('b')

    a._set_val(1)
    b._set_val(2)

    f = tan(tanh(a) * log(b))
    f.forward_pass()
    f.reverse(1, 1)
    reverse_grads = {
        'a': a.der,
        'b': b.der,
    }

    actual_f_val = np.tan(np.tanh(a.val) * np.log(b.val))
    actual_f_grad = {
        'a': ((1/np.cosh(a.val))**2) * np.log(b.val) * ((1 / np.cos(np.tanh(a.val) * np.log(b.val))) ** 2),
        'b': np.tanh(a.val) * ((1 / np.cos(np.tanh(a.val) * np.log(b.val))) ** 2) / b.val,
    }
    assert f.val == actual_f_val
    for var in actual_f_grad.keys():
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_sqrt():
    a = valNode('a')
    b = valNode('b')

    a._set_val(1)
    b._set_val(2)

    f = sqrt(tan(tanh(a) * log(b)))
    f.forward_pass()
    f.reverse(1, 1)
    reverse_grads = {
        'a': a.der,
        'b': b.der,
    }

    actual_f_val = np.sqrt(np.tan(np.tanh(a.val) * np.log(b.val)))
    actual_f_grad = {
        'a': (((1 / np.cosh(a.val)) ** 2) * np.log(b.val) * ((1 / np.cos(np.tanh(a.val)
                                                                         * np.log(b.val))) ** 2))/(2*actual_f_val),
        'b': np.tanh(a.val) * ((1 / np.cos(np.tanh(a.val) * np.log(b.val))) ** 2) / (b.val*actual_f_val*2),
    }
    assert f.val == actual_f_val
    for var in actual_f_grad.keys():
        assert np.isclose(actual_f_grad[var], reverse_grads[var])

def test_chain_rule():
    a = valNode('a')
    a._set_val(1.731)

    f = exp(cos(log(2 * (a + 5))))
    f.forward_pass()
    f.reverse(1, 1)
    reverse_grads = {
        'a': a.der,
    }
    actual_f_val = np.exp(np.cos(np.log(2 * (a.val + 5))))
    actual_f_grad = {
        'a': np.exp(np.cos(np.log(2 * (a.val + 5)))) * (-np.sin(np.log(2 * (a.val + 5)))) * (1 / (2 * (a.val + 5))) * (
            2)
    }
    assert f.val == actual_f_val
    for var in actual_f_grad.keys():
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_complex_func1():
    a = valNode('a')
    a._set_val(np.pi / 4)

    b = valNode('b')
    b._set_val(np.exp(1))

    f = sin(tan(a) / log(b))
    f.forward_pass()
    f.reverse(1, 1)
    actual_f_val = np.sin(np.tan(a.val) / np.log(b.val))
    assert f.val == actual_f_val
    actual_f_grad = {
        'a': ((np.cos(a.val)) ** (-2)) * (np.cos(np.tan(a.val) / np.log(b.val))) * 1 / np.log(b.val),
        'b': -1 * np.tan(a.val) * np.cos(np.tan(a.val) / np.log(b.val)) * (1 / b.val) * ((np.log(b.val)) ** (-2))
    }

    reverse_grads = {
        'a': a.der,
        'b': b.der,
    }
    for var in actual_f_grad.keys():
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_complex_func2():
    a = valNode('a')
    a._set_val(1)

    b = valNode('b')
    b._set_val(2)

    f = (a + b) ** 2 + log(a) * sin(b)
    f.forward_pass()
    f.reverse(1, 1)
    actual_f_val = (a.val + b.val) ** 2 + np.sin(b.val) * np.log(a.val)
    assert f.val == actual_f_val
    actual_f_grad = {
        'a': 2 * (a.val + b.val) + np.sin(b.val) / a.val,
        'b': 2 * (a.val + b.val) + np.log(a.val) * np.cos(b.val)
    }
    reverse_grads = {
        'a': a.der,
        'b': b.der,
    }
    for var in actual_f_grad.keys():
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


