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


def test_1d_ctor():
    m = MapArray(ndim=1)
    assert_equal(m.ndim, 1)
    assert_equal(m.shape, (0,))

    m[1] = 3
    assert_equal(m.shape, (2,))
    assert_equal(m[1], 3)


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


def test_to_coo():
    m = MapArray(ndim=4)
    m[0, 0, 0, 0] = 21.
    m[1, 2, 3, 4] = 22.
    m[2, 3, 4, 5] = 23.

    data, (i0, i1, i2, i3) = m.to_coo()
    assert_allclose(data, [21., 22., 23.], atol=1e-15)
    assert_equal(i0, [0, 1, 2])
    assert_equal(i1, [0, 2, 3])
    assert_equal(i2, [0, 3, 4])
    assert_equal(i3, [0, 4, 5])


if __name__ == "__main__":
    run_module_suite()
