import pytest
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")
from node import sin, cos, tan, exp, ln, valNode

# Anna's addition:
def test_cos_log():

    print("testing cos(ab/c) + c*log(a)")
    a = valNode('a')
    b = valNode('b')
    c = valNode('c')
    f = cos((a * b) / c) + c * log(a)
    a._set_val(4)
    b._set_val(-1)
    c._set_val(10)
    f_val, f_grad = f.forward()
    actual_f_val = np.cos((4 * -1) / 10) + 10 * np.log(4)
    actual_f_grad = {
        'a': (c.val / a.val) - b.val * np.sin((a.val * b.val) / c.val) / c.val,
        'b': (-a.val * np.sin((a.val * b.val) / c.val)) / c.val,
        'c': ((a.val * b.val * np.sin((a.val * b.val) / c.val) / (c.val ** 2)) + np.log(a.val))
    }

    assert np.isclose(f_val, actual_f_val)

    for var in f_grad:
        assert np.isclose(f_grad[var], actual_f_grad[var])

def test_cos_unit1():
    print("unit tests cos:")
    a = valNode('a')
    f = cos(a)
    a._set_val(np.pi / 4)
    f_val, f_grad = f.forward()

    assert np.isclose(f_val, np.cos(a.val))
    assert np.isclose(f_grad['a'], -np.sin(a.val))

def test_cos_unit2():
    print("unit tests cos:")
    a = valNode('a')
    f = cos(a)
    a._set_val(-10.0)
    f_val, f_grad = f.forward()

    assert np.isclose(f_val, np.cos(a.val))
    assert np.isclose(f_grad['a'], -np.sin(a.val))

def test_cos_const():
    f = cos(3)
    f_val, f_grad = f.forward()
    assert np.isclose(f_val, np.cos(3))
    assert np.isclose(f_grad[None], 0)

    f = cos(3.1)
    f_val, f_grad = f.forward()
    assert np.isclose(f_val, np.cos(3.1))
    assert np.isclose(f_grad[None], 0)

def test_cos_type_error():
    with pytest.raises(TypeError):
        cos('a')

def test_sin_unit1():
    print("unit tests sin:")
    a = valNode('a')
    f = sin(a)
    a._set_val(-np.pi / 3)
    f_val, f_grad = f.forward()

    assert np.isclose(f_val, np.sin(a.val))
    assert np.isclose(f_grad['a'], np.cos(a.val))

def test_sin_unit2():
    print("unit tests sin:")
    a = valNode('a')
    f = sin(a)
    a._set_val(11.0)
    f_val, f_grad = f.forward()

    assert np.isclose(f_val, np.sin(a.val))
    assert np.isclose(f_grad['a'], np.cos(a.val))

def test_sin_const():
    f = sin(3)
    f_val, f_grad = f.forward()
    assert np.isclose(f_val, np.sin(3))
    assert np.isclose(f_grad[None], 0)

    f = sin(3.1)
    f_val, f_grad = f.forward()
    assert np.isclose(f_val, np.sin(3.1))
    assert np.isclose(f_grad[None], 0)

def test_sin_type_error():
    with pytest.raises(TypeError):
        sin('a')

def test_tan_unit1():
    print("unit tests tan:")
    a = valNode('a')
    f = tan(a)
    a._set_val(np.pi/6)
    f_val, f_grad = f.forward()

    assert np.isclose(f_val, np.tan(a.val))
    assert np.isclose(f_grad['a'], 4/3)

def test_tan_unit2():
    print("unit tests tan:")
    a = valNode('a')
    f = tan(a)
    a._set_val(-2.5)
    f_val, f_grad = f.forward()

    assert np.isclose(f_val, np.tan(a.val))
    assert np.isclose(f_grad['a'], 1.55804)

def test_tan_const():
    f = tan(3)
    f_val, f_grad = f.forward()
    assert np.isclose(f_val, np.tan(3))
    assert np.isclose(f_grad[None], 0)

    f = tan(3.1)
    f_val, f_grad = f.forward()
    assert np.isclose(f_val, np.tan(3.1))
    assert np.isclose(f_grad[None], 0)

def test_tan_type_error():
    with pytest.raises(TypeError):
        tan('a')

def test_power_unit1():
    print("unit tests power:")
    a = valNode('a')
    f = a ** 3
    a._set_val(-5)
    f_val, f_grad = f.forward()

    assert np.isclose(f_val, a.val ** 3)
    assert np.isclose(f_grad['a'], 3 * a.val ** 2)

def test_power_unit2():
    print("unit tests power:")
    a = valNode('a')
    f = a ** (1 / 2)
    a._set_val(6.5)
    f_val, f_grad = f.forward()
    assert np.isclose(f_val, a.val ** (1 / 2))
    assert np.isclose(f_grad['a'], (1 / 2) * (a.val) ** (-1 / 2))

def test_exp_unit1():
    a = valNode('a')
    a._set_val(-5)
    f = exp(a)
    f_val, f_grad = f.forward()
    assert np.isclose(f_val, np.exp(a.val))
    assert np.isclose(f_grad['a'], np.exp(a.val))

def test_exp_const():
    f = exp(3)
    f_val, f_grad = f.forward()
    assert np.isclose(f_val, np.exp(3))
    assert np.isclose(f_grad[None], 0)

    f = exp(3.1)
    f_val, f_grad = f.forward()
    assert np.isclose(f_val, np.exp(3.1))
    assert np.isclose(f_grad[None], 0)

def test_exp_type_error():
    with pytest.raises(TypeError):
        exp('a')

def test_log_unit1():
    print("unit tests log:")
    a = valNode('a')
    f = log(a)
    a._set_val(10 / 3)
    f_val, f_grad = f.forward()
    assert np.isclose(f_val, np.log(a.val))
    assert np.isclose(f_grad['a'], 1 / a.val)

def test_log_unit2():
    print("unit tests log:")
    a = valNode('a')
    f = log(a)
    a._set_val(np.pi)
    f_val, f_grad = f.forward()
    assert np.isclose(f_val, np.log(a.val))
    assert np.isclose(f_grad['a'], 1 / a.val)

def test_log_const():
    f = log(10)
    f_val, f_grad = f.forward()
    assert np.isclose(f_val, np.log(10))
    assert np.isclose(f_grad[None], 0)

    f = log(10.1)
    f_val, f_grad = f.forward()
    assert np.isclose(f_val, np.log(10.1))
    assert np.isclose(f_grad[None], 0)

def test_log_type_error():
    with pytest.raises(TypeError):
        log('a')

def test_cos_log_multivar():
    print("testing cos(ab/c) + c*log(a)")
    a = valNode('a')
    b = valNode('b')
    c = valNode('c')
    f = cos((a * b) / c) + c * log(a)
    a._set_val(4)
    b._set_val(-1)
    c._set_val(10)
    f_val, f_grad = f.forward()
    actual_f_val = np.cos((4 * -1) / 10) + 10 * np.log(4)
    actual_f_grad = {
        'a': (c.val / a.val) - b.val * np.sin((a.val * b.val) / c.val) / c.val,
        'b': (-a.val * np.sin((a.val * b.val) / c.val)) / c.val,
        'c': ((a.val * b.val * np.sin((a.val * b.val) / c.val) / (c.val ** 2)) + np.log(a.val))
    }

    assert np.isclose(f_val, actual_f_val)

    for var in f_grad:
        assert np.isclose(f_grad[var], actual_f_grad[var])


# original tests:
def test_sin_multivar():
    print("testing: sin(ab + b) + c^a")
    a = valNode('a')
    b = valNode('b')
    c = valNode('c')
    f = sin(a * b + b) + c**a
    a._set_val(2)
    b._set_val(5)
    c._set_val(3)

    actual_f_val  = np.sin(a.val * b.val + b.val) + (c.val) ** a.val
    actual_f_grad = {
        'a': b.val * np.cos((a.val + 1) * b.val) + (c.val) ** (a.val) * np.log(c.val), 
        'b': (a.val + 1) * np.cos((a.val + 1) * b.val),
        'c': (a.val) * (c.val) ** (a.val - 1)
    }
    f.forward_pass()
    # print(f.val, )
    f.reverse(1,1)
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

    actual_f_val  = np.sin(a.val * b.val + b.val)
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
    f.reverse(1,1)
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
    a._set_val(np.pi/2)
    b._set_val(np.pi/3)
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
    f.reverse(1,1)
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
    c._set_val(np.pi/4)
    d._set_val(3)

    f = d * ln(a) + ln(exp(b)) + ln(cos(c))
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

    f = tan(exp(a/b))
    f.forward_pass()
    f.reverse(1, 1)
    reverse_grads = {
        'a': a.der,
        'b': b.der,
    }

    actual_f_val = np.tan(np.exp(a.val/b.val))
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

    f = a**2 + 3**a + 5/a + a/2
    f.forward_pass()
    f.reverse(1, 1)
    reverse_grads = {
        'a': a.der,
    }
    actual_f_val = a.val**2 + 3**a.val + 5/a.val + a.val/2
    actual_f_grad = {
        'a': 2 * a.val + 3**a.val * np.log(3) - 5 / (a.val**2) + 1/2
    }
    assert f.val == actual_f_val
    for var in actual_f_grad.keys():
        # print(actual_f_grad[var], reverse_grads[var])
        assert np.isclose(actual_f_grad[var], reverse_grads[var])
 
def test_chain_rule():
    a = valNode('a')
    a._set_val(1.731)

    f = exp(cos(ln(2*(a + 5))))
    f.forward_pass()
    f.reverse(1, 1)
    reverse_grads = {
        'a': a.der,
    }
    actual_f_val = np.exp(np.cos(np.log(2*(a.val + 5))))
    actual_f_grad = {
        'a': np.exp(np.cos(np.log(2*(a.val + 5)))) * (-np.sin(np.log(2*(a.val + 5)))) * (1 / (2*(a.val + 5))) * (2)
    }
    assert f.val == actual_f_val
    for var in actual_f_grad.keys():
        # print(actual_f_grad[var], reverse_grads[var])
        assert np.isclose(actual_f_grad[var], reverse_grads[var])
