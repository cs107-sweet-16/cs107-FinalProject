import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")
# from dualnumber import Dualnumber

def test_sample():
    assert True == True

# def add_dualnumbers():
    # # test passing 2 dual numbers to add
    # x = Dualnumber(1)
    # y = Dualnumber(2)
    # z = x + y
    # assert z.val == 3
    # assert z.der == 2

if __name__ == '__main__':
    test_sample()

