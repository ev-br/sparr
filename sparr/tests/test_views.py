from __future__ import division, print_function, absolute_import

import operator
from sys import getrefcount as refc

import numpy as np
from numpy.testing import (run_module_suite, TestCase, assert_equal, assert_,
                           assert_allclose, assert_raises,)
from numpy.testing.decorators import skipif, knownfailureif as knf

from .. import MapArray

from .test_slices import SLICES
from .test_basic import ArithmeticsMixin, CmpMixin


def test_view_stub():
    m = MapArray()
    m[1, 1] = 2.
    r0 = refc(m)

    mm = m[...]
    assert_(refc(mm) == r0)
    assert_(refc(m) == r0 + 1)

    del m
    assert_equal(refc(mm), r0)
    assert_allclose(mm.todense(), np.array([[0, 0], [0, 2]]), atol=1e-15)


def test_view_stub_writeable():
    # check that updates to the view propagate to the base
    m = MapArray()
    m[1, 1] = 2.

    mm = m[...]
    mm[0, 1] = 3.

    assert_allclose(m.todense(), np.array([[0, 3], [0, 2]]), atol=1e-15)

    del m
    assert_allclose(mm.todense(), np.array([[0, 3], [0, 2]]), atol=1e-15)


def test_view_stub_write_to_base():
    # check that updates to the base propagate to the view
    m = MapArray()
    m[1, 2] = 2.
    mm = m[...]

    m[0, 0] = 1.
    assert_allclose(m.todense(), mm.todense(), atol=1e-15)


@knf(True, "changing the shape of the base screws up the view")
def test_view_stub_write_to_base_2():
    # check that updates to the base propagate to the view
    m = MapArray()
    m[1, 2] = 2.
    mm = m[...]

    m[3, 4] = 1.
    assert_allclose(m.todense(), mm.todense(), atol=1e-15)


def test_copy():
    # copy of the view has no base
    m = MapArray()
    m[1, 2] = 2.
    mm = m[...]
    mmm = mm.copy()
    assert_(mmm.base is None)


def test_copy_shape():
    # check that view.copy() == view, not view.base 
    arr = np.arange(18).reshape((3, 6))
    m = MapArray.from_dense(arr)
    v = m[:-1, ::2]
    c = v.copy()
    assert_equal(v.shape, c.shape)
    assert_allclose(v.todense(), c.todense(), atol=1e-15)

    cc = v.astype(int)
    assert_equal(v.shape, cc.shape)
    assert_allclose(v.todense(), cc.todense(), atol=1e-15)


def test_getitem_slices():
    m = MapArray()
    m[0, 0] = 8
    m[11, 7] = 9
    m[8, 1] = 10

    slice_list = SLICES + [Ellipsis, 1]

    for s1 in slice_list:
        for s2 in slice_list:
            if s1 == s2 == Ellipsis:
                continue
            # in numpy, a[:, 1] has shape (12,) while a[:, 1:2] has shape (12, 1)
            fails = isinstance(s1, int) or isinstance(s2, int)
            yield knf(fails, "integer indices fail")(check_slicing), m, s1, s2


def check_slicing(m, s1, s2):
    mm = m[s1, s2]
    assert_allclose(mm.todense(), m.todense()[s1, s2], atol=1e-15)
    assert_(mm.base is m)
    assert_(m.base is None)

    mmm = mm[::2, :]
    assert_allclose(mmm.todense(), m.todense()[s1, s2][::2, :], atol=1e-15)
    assert_(mmm.base is mm)


def test_two_ellipsis():
    # a[..., 2, ...] is deprecated in numpy; we do not allow it either
    a = MapArray(shape=(2, 3, 4))
    with assert_raises(IndexError):
        a[..., 1, ...]



############### Test arithmetic operations with views   ###################

class ArithmViewsMixin(ArithmeticsMixin):
    def setUp(self):
        rndm = np.random.RandomState(1234)
        arr = rndm.random_sample(size=(2, 4)) * 10
        arr = arr.astype(self.dtype)
        ma = MapArray.from_dense(arr)
        ma[2, 4] = 1
        self.ma = ma[::2, ::-1]

        arr1 = rndm.random_sample(size=(3, 5)) * 10
        arr1 = arr1.astype(self.dtype)
        rhs = MapArray.from_dense(arr1)
        rhs.fill_value = 8
        self.rhs = rhs[::2, ::-1]

    @knf(True, "view shape update bug.")
    def test_inplace_iop_wrong_shape(self):
        # incompatible shapes should raise ValueErrors
        super(ArithmViewsMixin, self).test_inplace_iop_wrong_shape()

    @knf(True, "view shape update bug.")
    def test_sparse_op_sparse_wrong_shape(self):
        super(ArithmViewsMixin, self).test_sparse_op_sparse_wrong_shape()


class TestArithmViews(ArithmViewsMixin, TestCase):
    dtype = float


####### Test array[slices][slices]
class ArithmViewsMixin2(ArithmeticsMixin):
    def setUp(self):
        rndm = np.random.RandomState(1234)
        arr = rndm.random_sample(size=(2, 4)) * 10
        arr = arr.astype(self.dtype)
        ma = MapArray.from_dense(arr)
        ma[2, 4] = 1
        maa = ma[::2, ::-1]
        self.ma = maa[:, ::2]

        arr1 = rndm.random_sample(size=(3, 5)) * 10
        arr1 = arr1.astype(self.dtype)
        rhs = MapArray.from_dense(arr1)
        rhs.fill_value = 8
        rhss = rhs[::2, ::-1]
        self.rhs = rhss[:, ::2]

    @knf(True, "view shape update bug.")
    def test_inplace_iop_wrong_shape(self):
        # incompatible shapes should raise ValueErrors
        super(ArithmViewsMixin2, self).test_inplace_iop_wrong_shape()

    @knf(True, "view shape update bug.")
    def test_sparse_op_sparse_wrong_shape(self):
        super(ArithmViewsMixin2, self).test_sparse_op_sparse_wrong_shape()


class TestArithmViews2(ArithmViewsMixin2, TestCase):
    dtype = int


####################### Comparisons
class CmpViewsMixin(CmpMixin):

    def setUp(self):
        rndm = np.random.RandomState(1234)
        arr = rndm.random_sample(size=(3, 5)) * 10
        arr = arr.astype(self.ldtype)
        lhs = MapArray.from_dense(arr)
        self.lhs = lhs[::2, 1::2]

        arr1 = rndm.random_sample(size=(3, 5)) * 10 + 5
        arr1 = arr1.astype(self.rdtype)
        rhs = MapArray.from_dense(arr1)
        self.rhs = rhs[::2, 1::2]
        self.rhs.fill_value = -5


class TestCmpViews(CmpViewsMixin):
    ldtype = float
    rdtype = int

