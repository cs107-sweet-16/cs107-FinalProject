import pytest
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")
from dualnumber import Dualnumber

def test_sample():
    assert True == True

def test_add_dualnumbers():
    # test passing 2 dual numbers to add
    x = Dualnumber(1)
    y = Dualnumber(2)
    z = x + y
    assert z.val == 3
    assert z.der == 2

def test_add_dualnumber_and_nondual_number():
    # test passing 1 dual and 1 non dual number to add
    x = Dualnumber(4)
    y = 5
    z = x + y
    z_rev = y + x
    assert z.val == 9
    assert z.der == 1
    assert z_rev.val == 9
    assert z_rev.der == 1

def test_add_nondual_numbers():
    # test passing two real numbers to add
    x = 2
    y = 3
    z = x + y
    assert z == 5

def test_sub_dualnumbers():
    # test passing 2 dual numbers to subtract
    x = Dualnumber(1)
    y = Dualnumber(2)
    z = x - y
    assert z.val == -1
    assert z.der == 0

def test_sub_dualnumber_and_nondual_number():
    # test passing 1 dual and 1 non dual number to subtract
    x = Dualnumber(2)
    x.set_dual(4)
    y = 5
    z = x - y
    z_inv = y - x
    assert z.val == -3
    assert z.der == 4
    assert z_inv.val == 3
    assert z_inv.der == -4

def test_multiply_dualnumbers():
    # test passing two dual numbers to multiply
    x = Dualnumber(3)
    x.set_dual(4)
    y = Dualnumber(2)
    y.set_dual(5)
    z = x * y
    assert z.val == 6
    assert z.der == 23

def test_multiply_dualnumber_and_nondual_number():
    x = Dualnumber(3)
    x.set_dual(4)
    y = 5
    z = x * y
    z_inv = y * x
    assert z.val == 15
    assert z.der == 20
    assert z_inv.val == 15
    assert z_inv.der == 20

def test_divide_dualnumbers():
    x = Dualnumber(3)
    x.set_dual(4)
    y = Dualnumber(5)
    y.set_dual(3)
    z = x / y
    assert z.val == .6
    assert z.der == .44

def test_divide_dualnumber_and_nondual_number():
    x = Dualnumber(5)
    x.set_dual(4)
    y = 5
    z = x / y
    z_inv = y / x
    assert z.val == 1
    assert z.der == .8
    assert z_inv.val == 1
    assert z_inv.der == -.8

def test_pow_dualnumbers():
    # test passing 2 dual numbers through power:
    x = Dualnumber(2)
    x.set_dual(3)
    y = Dualnumber(4, der=5)

    z = x ** y
    assert z.val == 16
    assert np.isclose(z.der, 151.45177444479563)

def test_pow_dualnumber_and_nondual_number():
    # test passing 1 dual number and 1 int through power:
    x = Dualnumber(2, der=4)
    y = 3
    z = x ** y
    z_inv = y**x
    assert z.val == 8
    assert z.der == 48
    assert z_inv.val == 9
    assert np.isclose(z_inv.der, 4*np.log(3)*9)

def test_divide_dualnumbers_2():
    # test passing 2 dual numbers through division:
    x = Dualnumber(4)
    y = Dualnumber(2)
    z = x / y
    assert z.val == 2
    assert z.der == -0.5

def test_divide_dualnumber_and_nondual_number_2():
    # test passing 1 dual number and 1 int through division:
    x = Dualnumber(4)
    y = 2
    z = x / y
    assert z.val == 2
    assert z.der == 0.5

def test_neg_dualnumber():
    # test negating dual number
    x = Dualnumber(4)
    x.set_dual(4)
    x = -x
    assert x.val == -4 
    assert x.der == -4

def test_pos_dualnumber():
    # test positive of dual number
    x = Dualnumber(4)
    x.set_dual(4)
    assert x.val == 4 
    assert x.der == 4

def test_eq_dualnumber():
    # test equality of dual number
    x = Dualnumber(4)
    x.set_dual(4)

    y = Dualnumber(4)
    y.set_dual(4)

    assert x == y
