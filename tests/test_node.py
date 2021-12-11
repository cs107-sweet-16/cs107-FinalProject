import pytest
import sys
import os
import numpy as np

# sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from autodiff.node import sin, cos, tan, exp, ln, log, sinh, cosh, tanh, sqrt, logistic, log_ab, arcsin, arccos, \
    arctan, valNode, vector, variables


def test_pos():
    a = valNode('a')
    a._set_val(1)
    f = +a
    f_val, f_grad = f.forward()
    actual_f_val = (a.val)
    actual_f_grad = {
        'a': 1,
        None: 0,
    }

    assert np.isclose(f_val, actual_f_val)

    for var in f_grad:
        assert np.isclose(f_grad[var], actual_f_grad[var])


def test_neg():
    a = valNode('a')
    a._set_val(1)
    f = -a
    f_val, f_grad = f.forward()
    actual_f_val = -(a.val)
    actual_f_grad = {
        'a': -1,
        None: 0,
    }

    assert np.isclose(f_val, actual_f_val)

    for var in f_grad:
        assert np.isclose(f_grad[var], actual_f_grad[var])


def test_sub_int():
    a = valNode('a')
    a._set_val(1)
    f = a - 5
    f_val, f_grad = f.forward()
    actual_f_val = a.val - 5
    actual_f_grad = {
        'a': 1,
        None: 0,
    }

    assert np.isclose(f_val, actual_f_val)

    for var in f_grad:
        assert np.isclose(f_grad[var], actual_f_grad[var])


def test_rsub_int():
    a = valNode('a')
    a._set_val(1)
    f = 5 - a
    f_val, f_grad = f.forward()
    actual_f_val = 5 - a.val
    actual_f_grad = {
        'a': -1,
        None: 0,
    }

    assert np.isclose(f_val, actual_f_val)

    for var in f_grad:
        assert np.isclose(f_grad[var], actual_f_grad[var])


def test_rsub_float():
    a = valNode('a')
    a._set_val(1)
    f = a - 5.1
    f_val, f_grad = f.forward()
    actual_f_val = a.val - 5.1
    actual_f_grad = {
        'a': 1,
        None: 0,
    }

    assert np.isclose(f_val, actual_f_val)

    for var in f_grad:
        assert np.isclose(f_grad[var], actual_f_grad[var])


def test_rsub_float():
    a = valNode('a')
    a._set_val(1)
    f = 5.1 - a
    f_val, f_grad = f.forward()
    actual_f_val = 5.1 - a.val
    actual_f_grad = {
        'a': -1,
        None: 0,
    }

    assert np.isclose(f_val, actual_f_val)

    for var in f_grad:
        assert np.isclose(f_grad[var], actual_f_grad[var])


def test_cos_log():
    print("testing cos(ab/c) + c*log(a)")
    a = valNode('a')
    b = valNode('b')
    c = valNode('c')
    f = cos((a * b) / c) + c * ln(a)
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
    assert len(f_grad)==0

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
    a._set_val(np.pi / 6)
    f_val, f_grad = f.forward()

    assert np.isclose(f_val, np.tan(a.val))
    assert np.isclose(f_grad['a'], 4 / 3)


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
    a._set_val(5)
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


def test_power_type_error():
    with pytest.raises(TypeError):
        a = valNode('a')
        a._set_val("2")
        f = a ** (1 / 2)
        f.forward()


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
    f = ln(a)
    a._set_val(10 / 3)
    f_val, f_grad = f.forward()
    assert np.isclose(f_val, np.log(a.val))
    assert np.isclose(f_grad['a'], 1 / a.val)


def test_log_unit2():
    print("unit tests log:")
    a = valNode('a')
    f = ln(a)
    a._set_val(np.pi)
    f_val, f_grad = f.forward()
    assert np.isclose(f_val, np.log(a.val))
    assert np.isclose(f_grad['a'], 1 / a.val)


def test_log_const():
    f = ln(10)
    f_val, f_grad = f.forward()
    assert np.isclose(f_val, np.log(10))
    assert np.isclose(f_grad[None], 0)

    f = ln(10.1)
    f_val, f_grad = f.forward()
    assert np.isclose(f_val, np.log(10.1))
    assert np.isclose(f_grad[None], 0)


def test_log_type_error():
    with pytest.raises(TypeError):
        ln('a')


def test_cos_log_multivar():
    print("testing cos(ab/c) + c*log(a)")
    a = valNode('a')
    b = valNode('b')
    c = valNode('c')
    f = cos((a * b) / c) + c * ln(a)
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
    f.forward()
    # print(f.val, )
    f.reverse()
    reverse_grads = {
        'a': a.der,
        'b': b.der,
        'c': c.der,
    }
    assert f.val == actual_f_val
    for var in actual_f_grad:
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_sin_multivar_2():
    a = valNode('a')
    b = valNode('b')
    f = sin(a * b + b)
    a._set_val(2)
    b._set_val(5)

    f_val, f_grad = f.forward()

    actual_f_val = np.sin(a.val * b.val + b.val)
    actual_f_grad = {
        'a': b.val * np.cos((a.val + 1) * b.val),
        'b': (a.val + 1) * np.cos((a.val + 1) * b.val)
    }

    assert np.isclose(f_val, actual_f_val)

    for var in f_grad:
        assert np.isclose(f_grad[var], actual_f_grad[var])
    #

    f.forward()
    f.reverse()
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

    actual_f_grad = {
        'a': (np.exp(a.val / c.val)) / (c.val),
        'b': 1,
        'c': -(a.val * np.exp(a.val / c.val)) / (c.val ** 2),
    }

    f.forward()
    f.reverse()
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
    f.forward()
    f.reverse()
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

    f = d * ln(a) + ln(exp(b)) + ln(cos(c))
    f.forward()
    f.reverse()
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
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_tan_multivar2():
    a = valNode('a')
    b = valNode('b')
    a._set_val(2)
    b._set_val(np.pi)

    f = tan(exp(a / b))
    f.forward()
    f.reverse()
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
    f.forward()
    f.reverse()
    reverse_grads = {
        'a': a.der,
    }
    actual_f_val = a.val ** 2 + 3 ** a.val + 5 / a.val + a.val / 2
    actual_f_grad = {
        'a': 2 * a.val + 3 ** a.val * np.log(3) - 5 / (a.val ** 2) + 1 / 2
    }
    assert f.val == actual_f_val
    for var in actual_f_grad.keys():
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_logistic():
    a = valNode('a')
    a._set_val(2)
    b = valNode('b')
    b._set_val(3)

    f = logistic(sin(a) + cos(b))
    f.forward()
    f.reverse()
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
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_logistic_const():
    f = logistic(10)
    f_val, f_grad = f.forward()
    assert np.isclose(f_val, 0.5 * (1 + np.tanh(0.5 * 10)))
    assert np.isclose(f_grad[None], 0)


def test_logistic_type_error():
    with pytest.raises(TypeError):
        logistic("a")


def test_log_ab_both_nodes():
    a = valNode('a')
    a._set_val(2)
    b = valNode('b')
    b._set_val(3)

    f = log_ab(a + b, a * b)
    f.forward()
    f.reverse()
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
    f.forward()
    f.reverse()
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
    f.forward()
    f.reverse()
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
    f.forward()
    f.reverse()
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


def test_log_ab_type_error1():
    with pytest.raises(TypeError):
        log_ab("a", 2)


def test_log_ab_type_error2():
    with pytest.raises(TypeError):
        log_ab(1, "2")


def test_log_ab_type_error2():
    with pytest.raises(TypeError):
        a = valNode("a")
        a.set_val(2)
        log_ab(a, "2")


def test_log_ab_type_error2():
    with pytest.raises(TypeError):
        a = valNode("a")
        a.set_val(2)
        log_ab("2", a)


def test_arcsin1():
    a = valNode('a')
    b = valNode('b')
    c = valNode('c')

    a._set_val(.5)
    b._set_val(2)
    c._set_val(np.pi / 4)

    f = log(arcsin(a)) + log(exp(b)) + log(cos(c))
    f.forward()
    f.reverse()
    reverse_grads = {
        'a': a.der,
        'b': b.der,
        'c': c.der,
    }

    actual_f_val = np.log(np.arcsin(a.val)) + np.log(np.exp(b.val)) + np.log(np.cos(c.val))
    actual_f_grad = {
        'a': 1 / (np.sqrt(1-a.val*a.val)*np.arcsin(a.val)),
        'b': 1,
        'c': -np.tan(c.val),
    }
    assert f.val == actual_f_val
    for var in actual_f_grad.keys():
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_arcsin_const():
    f = arcsin(0)
    f_val, f_grad = f.forward()
    assert np.isclose(f_val, np.arcsin(0))
    assert np.isclose(f_grad[None], 0)


def test_arcsin_type_error():
    with pytest.raises(TypeError):
        arcsin("a")


def test_arccos():
    a = valNode('a')
    b = valNode('b')
    c = valNode('c')

    a._set_val(.5)
    b._set_val(2)
    c._set_val(np.pi / 4)

    f = log(arccos(a)) + log(exp(b)) + log(cos(c))
    f.forward()
    f.reverse()
    reverse_grads = {
        'a': a.der,
        'b': b.der,
        'c': c.der,
    }

    actual_f_val = np.log(np.arccos(a.val)) + np.log(np.exp(b.val)) + np.log(np.cos(c.val))
    actual_f_grad = {
        'a': -1 / (np.sqrt(1-a.val*a.val)*np.arccos(a.val)),
        'b': 1,
        'c': -np.tan(c.val),
    }
    assert f.val == actual_f_val
    for var in actual_f_grad.keys():
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_arccos_const():
    f = arccos(0)
    f_val, f_grad = f.forward()
    assert np.isclose(f_val, np.arccos(0))
    assert np.isclose(f_grad[None], 0)


def test_arccos_type_error():
    with pytest.raises(TypeError):
        arccos("a")


def test_arctan():
    a = valNode('a')
    b = valNode('b')
    c = valNode('c')

    a._set_val(1)
    b._set_val(2)
    c._set_val(np.pi / 4)

    f = log(arctan(a)) + log(exp(b)) + log(cos(c))
    f.forward()
    f.reverse()
    reverse_grads = {
        'a': a.der,
        'b': b.der,
        'c': c.der,
    }

    actual_f_val = np.log(np.arctan(a.val)) + np.log(np.exp(b.val)) + np.log(np.cos(c.val))
    actual_f_grad = {
        'a': 1 / ((1+a.val*a.val)*np.arctan(a.val)),
        'b': 1,
        'c': -np.tan(c.val),
    }
    assert f.val == actual_f_val
    for var in actual_f_grad.keys():
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_arctan_const():
    f = arctan(1)
    f_val, f_grad = f.forward()
    assert np.isclose(f_val, np.arctan(1))
    assert np.isclose(f_grad[None], 0)


def test_arctan_type_error():
    with pytest.raises(TypeError):
        arctan("a")


def test_sinh1():
    a = valNode('a')
    b = valNode('b')
    c = valNode('c')

    a._set_val(1)
    b._set_val(2)
    c._set_val(np.pi / 4)

    f = log(sinh(a)) + log(exp(b)) + log(cos(c))
    f.forward()
    f.reverse()
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
    f.forward()
    f.reverse()
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


def test_sinh_const():
    f = sinh(10)
    f_val, f_grad = f.forward()
    assert np.isclose(f_val, np.sinh(10))
    assert np.isclose(f_grad[None], 0)


def test_sinh_type_error():
    with pytest.raises(TypeError):
        sinh("a")


def test_cosh1():
    a = valNode('a')
    b = valNode('b')
    c = valNode('c')

    a._set_val(1)
    b._set_val(2)
    c._set_val(np.pi / 4)

    f = log(cosh(a)) + log(exp(b)) + log(cos(c))
    f.forward()
    f.reverse()
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
    f.forward()
    f.reverse()
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


def test_cosh_const():
    f = cosh(10)
    f_val, f_grad = f.forward()
    assert np.isclose(f_val, np.cosh(10))
    assert np.isclose(f_grad[None], 0)


def test_cosh_type_error():
    with pytest.raises(TypeError):
        cosh("a")


def test_tanh1():
    a = valNode('a')
    b = valNode('b')
    c = valNode('c')

    a._set_val(1)
    b._set_val(2)
    c._set_val(np.pi / 4)

    f = log(tanh(a)) + log(exp(b)) + log(cos(c))
    f.forward()
    f.reverse()
    reverse_grads = {
        'a': a.der,
        'b': b.der,
        'c': c.der,
    }

    actual_f_val = np.log(np.tanh(a.val)) + np.log(np.exp(b.val)) + np.log(np.cos(c.val))
    actual_f_grad = {
        'a': 1 / (np.sinh(a.val) * np.cosh(a.val)),
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
    f.forward()
    f.reverse()
    reverse_grads = {
        'a': a.der,
        'b': b.der,
    }

    actual_f_val = np.tan(np.tanh(a.val) * np.log(b.val))
    actual_f_grad = {
        'a': ((1 / np.cosh(a.val)) ** 2) * np.log(b.val) * ((1 / np.cos(np.tanh(a.val) * np.log(b.val))) ** 2),
        'b': np.tanh(a.val) * ((1 / np.cos(np.tanh(a.val) * np.log(b.val))) ** 2) / b.val,
    }
    assert f.val == actual_f_val
    for var in actual_f_grad.keys():
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_tanh_const():
    f = tanh(10)
    f_val, f_grad = f.forward()
    assert np.isclose(f_val, np.tanh(10))
    assert np.isclose(f_grad[None], 0)


def test_tanh_type_error():
    with pytest.raises(TypeError):
        tanh("a")


def test_sqrt():
    a = valNode('a')
    b = valNode('b')

    a._set_val(1)
    b._set_val(2)

    f = sqrt(tan(tanh(a) * log(b)))
    f.forward()
    f.reverse()
    reverse_grads = {
        'a': a.der,
        'b': b.der,
    }

    actual_f_val = np.sqrt(np.tan(np.tanh(a.val) * np.log(b.val)))
    actual_f_grad = {
        'a': (((1 / np.cosh(a.val)) ** 2) * np.log(b.val) * ((1 / np.cos(np.tanh(a.val)
                                                                         * np.log(b.val))) ** 2)) / (2 * actual_f_val),
        'b': np.tanh(a.val) * ((1 / np.cos(np.tanh(a.val) * np.log(b.val))) ** 2) / (b.val * actual_f_val * 2),
    }
    assert f.val == actual_f_val
    for var in actual_f_grad.keys():
        assert np.isclose(actual_f_grad[var], reverse_grads[var])


def test_sqrt_const():
    f = sqrt(10)
    f_val, f_grad = f.forward()
    assert np.isclose(f_val, np.sqrt(10))
    assert np.isclose(f_grad[None], 0)


def test_sqrt_type_error():
    with pytest.raises(TypeError):
        sqrt("a")


def test_chain_rule():
    a = valNode('a')
    a._set_val(1.731)

    f = exp(cos(ln(2 * (a + 5))))
    f.forward()
    f.reverse()
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
    f.forward()
    f.reverse()
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
    f.forward()
    f.reverse()
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


def test_variables():
    a = variables('a')
    b = variables('b')
    a.set_val(2)
    b.set_val(3)
    f = (a + b) ** 2 + log(a) * sin(b)
    f.forward()
    f.reverse()
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


def test_vector_value_error():
    with pytest.raises(ValueError):
        v = variables('v', 2)
        v.set_val([2, 3, 5])


def test_vector_func_vector_input():
    v = variables('v', 2)
    v.set_val([2, 3])

    def func(v):
        """
            f takes a size=3 vector and output a size=2 vector
        """
        f1 = (v[0] + v[1]) ** 2 + log(v[0]) * sin(v[1])
        f2 = sin(tan(v[0]) / log(v[1]))
        return vector(f1, f2)

    f = func(v)
    res = f.evaluate()
    assert np.isclose((v[0].val + v[1].val) ** 2 + np.sin(v[1].val) * np.log(v[0].val), res[0])
    assert np.isclose((np.sin(np.tan(v[0].val) / np.log(v[1].val))), res[1])
    der_arr = f.grad(v)
    assert np.isclose(2 * (v[0].val + v[1].val) + np.sin(v[1].val) / v[0].val, der_arr[0][0])
    assert np.isclose(2 * (v[0].val + v[1].val) + np.log(v[0].val) * np.cos(v[1].val), der_arr[0][1])
    assert np.isclose(((np.cos(v[0].val)) ** (-2)) * (np.cos(np.tan(v[0].val) / np.log(v[1].val))
                                                      ) * 1 / np.log(v[1].val), der_arr[1][0])
    assert np.isclose(-1 * np.tan(v[0].val) * np.cos(np.tan(v[0].val) / np.log(v[1].val)) * (1 / v[1].val) * (
            (np.log(v[1].val)) ** (-2)), der_arr[1][1])


def test_vec_func_scalar_input():
    v = variables('v', 1)
    v.set_val(2)

    def func1(x):
        f1 = x ** 2
        f2 = x ** 3
        return vector(f1, f2)

    f = func1(v)
    res = f.evaluate()
    assert np.isclose(res[0], 4)
    assert np.isclose(res[1], 8)
    der_arr = f.grad(v)
    print(der_arr, res)
    assert np.isclose(der_arr[0], 4)
    assert np.isclose(der_arr[1], 12)


def test_vec_func_type_error1():
    with pytest.raises(TypeError):
        v = variables('v', 1)
        v.set_val([2])


def test_set_val_type_error():
    with pytest.raises(TypeError):
        v = valNode('v')
        v.set_val("2")
