from __future__ import division, print_function, absolute_import
import operator

import numpy as np
from numpy.testing import (run_module_suite, TestCase, assert_equal, assert_,
                           assert_allclose, assert_raises,)
from numpy.testing.decorators import knownfailureif, skipif

from sp_map import MapArray


class BasicMixin(object):
    def test_ctor(self):
        ma = MapArray(dtype=self.dtype)
        assert_equal(ma.dtype, self.dtype)
        assert_equal(ma.ndim, 2)
        assert_equal(ma.shape, (0, 0))
        assert_allclose(ma.fill_value, 0., atol=1e-15)

        ma = MapArray(shape=(3, 4), dtype=self.dtype)
        assert_equal(ma.todense().shape, (3, 4))

        ma = MapArray(shape=(3, 4), fill_value=42, dtype=self.dtype)
        assert_equal(ma.fill_value, self.dtype(42))

        with assert_raises(TypeError):
            MapArray(unknown='arg')

        with assert_raises(TypeError):
            MapArray(dtype=42)

    def test_fill_value_mutable(self):
        ma = MapArray(dtype=self.dtype)
        ma.fill_value = -1.
        assert_equal(ma.fill_value, self.dtype(-1))

        if self.dtype != np.dtype(bool):
            with assert_raises(TypeError):
                ma.fill_value = 'lalala'
        else:
            ma.fill_value = 'lalala'
            assert_equal(ma.fill_value, True)

    def test_basic_insert(self):
        ma = MapArray(dtype=self.dtype)
        ma[1, 1] = -1
        assert_equal(ma.shape, (2, 2))
        assert_equal(ma.ndim, 2)
        assert_equal(ma.count_nonzero(), 1)
        assert_allclose(ma.todense(),
                        np.array([[0, 0],
                                  [0, -1]], dtype=self.dtype), atol=1e-15)

        ma[2, 3] = 8
        assert_equal(ma.ndim, 2)
        assert_equal(ma.shape, (3, 4))
        assert_equal(ma.count_nonzero(), 2)
        assert_allclose(ma.todense(),
                        np.array([[0, 0, 0, 0],
                                  [0, -1, 0, 0],
                                  [0, 0, 0, 8]], dtype=self.dtype), atol=1e-15)

    def test_todense_fillvalue(self):
        ma = MapArray(dtype=self.dtype)
        ma[2, 3] = 8
        ma[1, 1] = -1
        ma.fill_value = 1
        assert_equal(ma.count_nonzero(), 2)
        assert_allclose(ma.todense(),
                        np.array([[1, 1, 1, 1],
                                  [1, -1, 1, 1],
                                  [1, 1, 1, 8]], dtype=self.dtype), atol=1e-15)

    @skipif(True, '32 bit indices')
    def test_int_overflow(self):
        # check array size s.t. a flat index overflows the C int range
        ma = MapArray()
        j = np.iinfo(np.int32).max + 1
        ma[1, j] = 1.
        ma.todense()

    def test_copy(self):
        ma = MapArray(dtype=self.dtype)
        ma[1, 1] = 1.

        # operate on a copy; the original must be intact
        ma1 = ma.copy()
        ma1[2, 4] = 3.

        assert_equal(ma1.dtype, self.dtype)
        assert_equal(ma1.shape, (3, 5))
        assert_allclose(ma1.todense(),
                        np.array([[0, 0, 0, 0, 0],
                                  [0, 1, 0, 0, 0], 
                                  [0, 0, 0, 0, 3]], dtype=self.dtype), atol=1e-15)
        assert_equal(ma.shape, (2, 2))
        assert_allclose(ma.todense(),
                        np.array([[0, 0],
                                  [0, 1]], dtype=self.dtype), atol=1e-15)


class TestBasicDouble(BasicMixin, TestCase):
    dtype = float


class TestBasicFloat(BasicMixin, TestCase):
    dtype = np.float32


class TestBasicPyInt(BasicMixin, TestCase):
    dtype = int


class TestBasicPyBool(BasicMixin, TestCase):
    dtype = bool


class ArithmeticsMixin(object):

    iop = operator.iadd       # x = iop(x, y) is x += y
    op = operator.add

    def setUp(self):
        ma = MapArray(dtype=self.dtype)
        ma[1, 1] = 1.
        ma[2, 4] = 2.
        self.ma = ma

        rhs = MapArray(dtype=self.dtype)
        rhs[2, 4] = 3.
        rhs.fill_value = -8
        self.rhs = rhs

    def test_inplace_iop_scalar(self):
        ma1 = self.ma.copy()
        ma1 = self.iop(ma1, 4)      # IOW, ma1 += 4

        assert_equal(ma1.shape, self.ma.shape)
        assert_allclose(ma1.todense(),
                        self.op(self.ma.todense(), 4.), atol=1e-15)

    def test_inplace_iop_unsupported_type_obj(self):
        ma1 = self.ma.copy()
        with assert_raises(TypeError):
            self.iop(ma1, 'lalala')

        with assert_raises(TypeError):
            self.iop(ma1, None)

        with assert_raises(TypeError):
            ress = self.iop(ma1, [1, 2, 3, 4])

    def test_inplace_iop_sparse(self):
        ma1 = self.ma.copy()
        rhs = self.rhs.copy()
        ma1 = self.iop(ma1, rhs)

        # the LHS is operated on, and RHS is intact
        assert_(isinstance(ma1, MapArray))
        assert_allclose(ma1.todense(),
                        self.op(self.ma.todense(), self.rhs.todense()), atol=1e-15)
        assert_allclose(rhs.todense(), self.rhs.todense(), atol=1e-15)

    def test_inplace_iop_wrong_shape(self):
        # incompatible shapes should raise ValueErrors
        ma1 = self.ma.copy()
        rhs = self.rhs.copy()
        rhs[8, 9] = -101
        with assert_raises(ValueError):
            self.iop(ma1, rhs)

    def test_sparse_op_sparse_wrong_shape(self):
        ma1 = self.ma.copy()
        rhs = self.rhs.copy()
        rhs[8, 9] = -101
        with assert_raises(ValueError):
            self.op(ma1, rhs)

    def test_sparse_op_sparse(self):
        ma1 = self.ma.copy()
        rhs = self.rhs.copy()

        res = self.op(ma1, rhs)
        assert_(isinstance(res, MapArray))
        assert_allclose(res.todense(),
                        self.op(ma1.todense(), rhs.todense()), atol=1e-15)

    def test_sparse_op_scalar(self):
        ma1 = self.ma.copy()

        res = self.op(ma1, 4)
        assert_(isinstance(res, MapArray))
        assert_allclose(res.todense(),
                        self.op(ma1.todense(), 4), atol=1e-15)

    def test_scalar_op_sparse(self):
        ma1 = self.ma.copy()

        res = self.op(4, ma1)
        assert_(isinstance(res, MapArray))
        assert_allclose(res.todense(),
                        self.op(4, ma1.todense()), atol=1e-15)

    def test_sparse_op_wrong_scalar(self):
        ma1 = self.ma.copy()
        with assert_raises(TypeError):
            self.op(ma1, 'lalala')
        with assert_raises(TypeError):
            self.op('lalala', ma1)

        with assert_raises(TypeError):
            self.op(ma1, None)
        with assert_raises(TypeError):
            self.op(None, ma1)

        with assert_raises(TypeError):
            self.iop(ma1, [1, 2, 3, 4])
        with assert_raises(TypeError):
            self.iop([1, 2, 3, 4], ma1)

    def test_sparse_dense_interop(self):
        # dense + sparse densifies for scipy.sparse matrices.
        # So we try to be consistent here for sparse + dense or sparse += dense
        ma1 = self.ma.copy()
        dense = self.rhs.todense()

        for res in (self.op(ma1, dense),
                    self.op(dense, ma1)):
            assert_(isinstance(res, np.ndarray))
            assert_allclose(res,
                            self.op(ma1.todense(), dense), atol=1e-15)

        # also check the in-place version
        ma1 = self.ma.copy()
        dense = self.rhs.todense()

        ma1 = self.iop(ma1, dense)
        assert_(isinstance(ma1, np.ndarray))
        assert_allclose(ma1,
                        self.op(self.ma.todense(), dense), atol=1e-15)

    def test_op_self(self):
        # check aliasing: op and iop with self in the r.h.s. should work OK
        ma = self.ma.copy()

        ma2 = self.op(ma, ma)
        assert_allclose(ma2.todense(),
                        self.op(ma.todense(), ma.todense()), atol=1e-15)
        assert_allclose(ma.todense(), self.ma.todense(), atol=1e-15)

        # now test iop:
        ma2 = self.ma.copy()
        ma2 = self.iop(ma2, ma2)
        assert_allclose(ma2.todense(),
                        self.op(ma.todense(), ma.todense()), atol=1e-15)


class TestArithmDouble(ArithmeticsMixin, TestCase):
    dtype = float


class TestArithmFloat(ArithmeticsMixin, TestCase):
    dtype = np.float32


class TestArithmPyInt(ArithmeticsMixin, TestCase):
    dtype = int


class TestArithmPyBool(ArithmeticsMixin, TestCase):
    dtype = bool


class MulMixin(ArithmeticsMixin):
    iop = operator.imul       # x = iop(x, y) is x *= y
    op = operator.mul


class TestMulDouble(MulMixin, TestCase):
    dtype = float


class TestMulFloat(MulMixin, TestCase):
    dtype = np.float32


class TestMulPyInt(MulMixin, TestCase):
    dtype = int


class TestMulPyBool(MulMixin, TestCase):
    dtype = bool



class TestCasting(TestCase):
    m = MapArray(dtype=float)
    m[1, 1] = 10.0

    im = MapArray(dtype=int)
    im[1, 1] = -1.

    def test_double_int(self):
        # double + int becomes double
        res = self.m + self.im
        assert_equal(res.dtype, self.m.dtype)
        assert_allclose(res.todense(), np.array([[0., 0.],
                                                 [0., 9.]]), atol=1e-15)

    def test_int_double(self):
        # int + double becomes double
        res = self.im + self.m
        assert_equal(res.dtype, self.m.dtype)
        assert_allclose(res.todense(), np.array([[0., 0.],
                                                 [0., 9.]]), atol=1e-15)

    def test_inplace_double_int(self):
        m = self.m.copy()
        im = self.im.copy()

        m += im

        assert_equal(m.dtype, self.m.dtype)
        assert_allclose(m.todense(), np.array([[0., 0.],
                                               [0., 9.]]), atol=1e-15)

    def test_scalar_upcast(self):
        # double scalar + int array: upcasts the array
        im = self.im.copy()

        m = im + 1.2
        assert_equal(m.dtype, np.dtype(float))
        assert_allclose(m.todense(), im.todense() + 1.2, atol=1e-15)

        m = 1.2 + im
        assert_equal(m.dtype, np.dtype(float))
        assert_allclose(m.todense(), im.todense() + 1.2, atol=1e-15)


if __name__ == "__main__":
    run_module_suite()
