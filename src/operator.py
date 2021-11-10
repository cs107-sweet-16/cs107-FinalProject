from dualnumber import Dualnumber
import numpy as np
from typing import Union


def sin(x: Union[float, Dualnumber]) -> Dualnumber:
    try:
        sin_x = Dualnumber(np.sin(x.val))
        sin_x.set_dual(np.cos(x.val) * x.der)
        return sin_x
    except AttributeError as e:
        sin_x = Dualnumber(np.sin(x.val))
        sin_x.set_dual(0)
        return sin_x


def cos(x: Union[float, Dualnumber]) -> Dualnumber:
    pass


def tan(x: Union[float, Dualnumber]) -> Dualnumber:
    pass


def exp(x: Union[float, Dualnumber]) -> Dualnumber:
    pass


def log(x: Union[float, Dualnumber]) -> Dualnumber:
    pass





