from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import (run_module_suite, TestCase, assert_equal, assert_,
                           assert_allclose, assert_raises,)
from numpy.testing.decorators import knownfailureif, skipif

from .. import MapArray

def test_3d_ctor():
    # look, ma, it's 3D
    m = MapArray(ndim=3)
    assert_equal(m.ndim, 3)
    assert_equal(m.shape, (0, 0, 0))

    with assert_raises(IndexError):
        m[1, 1]

    m[1, 1, 2] = 8
    assert_equal(m.shape, (2, 2, 3))


def test_todense():
    m = MapArray(shape=(2, 3, 4))
    assert_equal(m.ndim, 3)

    # these are all fill_value
    assert_allclose(m.todense(),
                    np.zeros((2, 3, 4)), atol=1e-15)

    # add a non-zero
    m[1, 1, 1] = 8
    dense = np.zeros((2, 3, 4))
    dense[1, 1, 1] = 8
    assert_allclose(m.todense(), dense, atol=1e-15)


def test_fromdense():
    rndm = np.random.RandomState(1223)
    arr = rndm.uniform(size=(2, 3, 4))
    arr[arr > 0.5] = 0.
    m = MapArray.from_dense(arr)

    assert_equal(m.ndim, 3)
    assert_equal(m.shape, (2, 3, 4))
    assert_allclose(m.todense(), arr, atol=1e-15)


def test_copy():
    rndm = np.random.RandomState(1234)
    arr = rndm.uniform(size=(2, 3, 4))
    arr[arr < 0.5] = 0.
    m = MapArray.from_dense(arr)
    m1 = m.copy()

    assert_allclose(arr, m1.todense(), atol=1e-15)


def test_binop():
    rndm = np.random.RandomState(1234)
    arr = rndm.uniform(size=(2, 3, 4))
    arr[arr < 0.5] = 0.
    m = MapArray.from_dense(arr)
    m1 = 2.* m.copy()

    m2 = m + m1
    assert_allclose(m2.todense(), arr*3., atol=1e-15)

    # now try with mismatching dimensions
    a = np.ones((1, 2))
    ma = MapArray.from_dense(a)

    with assert_raises(ValueError):
        m + ma


def test_comparisons():
    rndm = np.random.RandomState(1234)
    arr = rndm.uniform(size=(2, 3, 4))
    m = MapArray.from_dense(arr)

    arr1 = arr.copy()
    arr1[arr1 < 0.5] = 1.
    m1 = MapArray.from_dense(arr1)

    assert_equal((m < m1).todense(), arr < arr1)

    # and dimensions mismatch
    a = np.ones((1, 2))
    ma = MapArray.from_dense(a)
    
    with assert_raises(ValueError):
        m < ma


def test_high_dim():
    rndm = np.random.RandomState(1234)
    arr = rndm.uniform(size=(2, 4, 3, 2))
    m = MapArray.from_dense(arr)

    assert_equal(arr.shape, m.shape)
    assert_allclose(m.todense(), arr, atol=1e-15)


if __name__ == "__main__":
    run_module_suite()
