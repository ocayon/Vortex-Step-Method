from numba import jit
import numpy as np


@jit(nopython=True)
def jit_cross(a, b):
    return np.cross(a, b)


@jit(nopython=False)
def jit_norm(value):
    return np.linalg.norm(value)
