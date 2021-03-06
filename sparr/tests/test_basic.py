from __future__ import division, print_function, absolute_import
import operator

import numpy as np
from numpy.testing import (run_module_suite, TestCase, assert_equal, assert_,
                           assert_allclose, assert_raises,)
from numpy.testing.decorators import knownfailureif, skipif
try:
    from numpy.testing import SkipTest
except ImportError:
    from nose import SkipTest

from .. import MapArray

try:
    from scipy._lib._version import NumpyVersion
except ImportError:
    def NumpyVersion(vstr):
        return vstr
OLD_NUMPY = NumpyVersion(np.__version__) < '1.9.1'

try:
    from scipy import sparse
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


class BasicMixin(object):
    def test_ctor(self):
        ma = MapArray(dtype=self.dtype)
        assert_equal(ma.dtype, self.dtype)
        assert_equal(ma.ndim, 2)
        assert_equal(ma.shape, (0, 0))

        assert_allclose(ma.fill_value, self.dtype(0), atol=1e-15)

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
        ma.fill_value = 8
        assert_equal(ma.fill_value, self.dtype(8))

        if self.dtype != np.dtype(bool):
            with assert_raises(TypeError):
                ma.fill_value = 'lalala'
        else:
            ma.fill_value = 'lalala'
            assert_equal(ma.fill_value, True)

    def test_basic_insert(self):
        ma = MapArray(dtype=self.dtype)
        ma[1, 1] = 2
        assert_equal(ma.shape, (2, 2))
        assert_equal(ma.ndim, 2)
        assert_equal(ma.count_nonzero(), 1)
        assert_allclose(ma.todense(),
                        np.array([[0, 0],
                                  [0, 2]], dtype=self.dtype), atol=1e-15)

        ma[2, 3] = 8
        assert_equal(ma.ndim, 2)
        assert_equal(ma.shape, (3, 4))
        assert_equal(ma.count_nonzero(), 2)
        assert_allclose(ma.todense(),
                        np.array([[0, 0, 0, 0],
                                  [0, 2, 0, 0],
                                  [0, 0, 0, 8]], dtype=self.dtype), atol=1e-15)

    def test_shape_fixed(self):
        # if shape is set in ctor explicitly, it's fixed
        m = MapArray(dtype=self.dtype, shape=(3, 4))

        assert_equal(m.is_shape_fixed, True)
        with assert_raises(IndexError):
            m[4, 5] = 4
        assert_equal(m.shape, (3, 4))

        # can manually toggle the shape being mutable or not
        m.is_shape_fixed = False
        m[4, 5] = 4
        assert_equal(m[4, 5], self.dtype(4))
        assert_equal(m.shape, (5, 6))

    def test_todense_fillvalue(self):
        ma = MapArray(dtype=self.dtype)
        ma[2, 3] = 8
        ma[1, 1] = 2
        ma.fill_value = 1
        assert_equal(ma.count_nonzero(), 2)
        assert_allclose(ma.todense(),
                        np.array([[1, 1, 1, 1],
                                  [1, 2, 1, 1],
                                  [1, 1, 1, 8]], dtype=self.dtype), atol=1e-15)

    def test_int_overflow(self):
        # check array size s.t. a flat index overflows the C int range
        ma = MapArray()
        j = np.iinfo(np.int32).max + 1
        ma[1, j] = 1.
        assert_equal(ma.count_nonzero(), 1)
        assert_equal(ma.shape, (2, j+1))

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

    def test_from_dense(self):
        rndm = np.random.RandomState(1234)
        arr = rndm.random_sample(size=(2, 3)) * 10
        arr = arr.astype(self.dtype)

        m = MapArray.from_dense(arr)
        assert_equal(m.shape, arr.shape)
        assert_equal(m.dtype, arr.dtype)
        assert_allclose(m.todense(), arr, atol=1e-15)

    def test_co_coo(self):
        m = MapArray(dtype=self.dtype)
        m[1, 2] = self.dtype(11)
        m[0, 0] = self.dtype(22)

        data, (row, col) = m.to_coo()
        assert_equal(row, [0, 1])
        assert_equal(col, [0, 2])
        assert_equal(data, [self.dtype(22), self.dtype(11)])

    def test_to_coo_int32_overflow(self):
        ma = MapArray(dtype=self.dtype)
        j = np.iinfo(np.int32).max + 1
        ma[1, j] = self.dtype(1)
        data, (row, col) = ma.to_coo()
        assert_equal(row, [1])
        assert_equal(col, [j])
        assert_equal(data, self.dtype(1))

    @skipif(not HAVE_SCIPY)
    def test_coo_matrix(self):
        rndm = np.random.RandomState(122)
        a = rndm.random_sample(size=(8, 8))
        coom = sparse.coo_matrix(a)

        m = MapArray.from_dense(a)
        data, (row, col) = m.to_coo()
        assert_equal(data, coom.data)
        assert_equal(row, coom.row)
        assert_equal(col, coom.col)

    def test_from_coo(self):
        rndm = np.random.RandomState(122)
        a = rndm.random_sample(size=(4, 4))
        m = MapArray.from_dense(a)
        data, (row, col) = m.to_coo()

        m1 = MapArray.from_coo(data, (row, col))
        assert_equal(m, m1)


class TestBasicDouble(BasicMixin, TestCase):
    dtype = float


class TestBasicFloat(BasicMixin, TestCase):
    dtype = np.float32


class TestBasicPyInt(BasicMixin, TestCase):
    dtype = int


class TestBasicNpInt8(BasicMixin, TestCase):
    dtype = np.int8


class TestBasicNpInt16(BasicMixin, TestCase):
    dtype = np.int16


class TestBasicNpInt32(BasicMixin, TestCase):
    dtype = np.int32


class TestBasicNpInt64(BasicMixin, TestCase):
    dtype = np.int64


class TestBasicNpUInt8(BasicMixin, TestCase):
    dtype = np.uint8


class TestBasicNpUInt16(BasicMixin, TestCase):
    dtype = np.uint16


class TestBasicNpUInt32(BasicMixin, TestCase):
    dtype = np.uint32


class TestBasicNpUInt64(BasicMixin, TestCase):
    dtype = np.uint64


class TestBasicPyBool(BasicMixin, TestCase):
    dtype = bool


class TestBasicNpBool(BasicMixin, TestCase):
    dtype = np.bool


class TestBasicNpBool_(BasicMixin, TestCase):
    dtype = np.bool_


class TestBasicComplex(BasicMixin, TestCase):
    dtype = complex

############################ Arithmetic binops

class ArithmeticsMixin(object):

    iop = operator.iadd       # x = iop(x, y) is x += y
    op = operator.add

    def setUp(self):
        rndm = np.random.RandomState(1234)
        arr = rndm.random_sample(size=(2, 4)) * 10
        arr = arr.astype(self.dtype)
        self.ma = MapArray.from_dense(arr)
        self.ma[2, 4] = 1

        arr1 = rndm.random_sample(size=(3, 5)) * 10
        arr1 = arr1.astype(self.dtype)
        self.rhs = MapArray.from_dense(arr1)
        self.rhs.fill_value = 8

    def test_inplace_iop_scalar(self):
        ma1 = self.ma.copy()

        scalar = np.min(ma1.todense())
        ma1 = self.iop(ma1, scalar)      # IOW, ma1 += scalar

        assert_equal(ma1.shape, self.ma.shape)
        assert_allclose(ma1.todense(),
                        self.op(self.ma.todense(), scalar), atol=1e-15)

    def test_inplace_iop_unsupported_type_obj(self):
        ma1 = self.ma.copy()
        with assert_raises(TypeError):
            self.iop(ma1, 'lalala')

        with assert_raises(TypeError):
            self.iop(ma1, None)

        with assert_raises(TypeError):
            self.iop(ma1, [1, 2, 3, 4])

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
        rhs[8, 9] = 101
        with assert_raises(ValueError):
            self.iop(ma1, rhs)

    def test_sparse_op_sparse_wrong_shape(self):
        ma1 = self.ma.copy()
        rhs = self.rhs.copy()
        rhs[8, 9] = 101
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
        scalar = np.min(ma1.todense())

        res = self.op(ma1, scalar)
        assert_(isinstance(res, MapArray))
        assert_allclose(res.todense(),
                        self.op(ma1.todense(), scalar), atol=1e-15)

    def test_scalar_op_sparse(self):
        ma1 = self.ma.copy()
        scalar = np.min(ma1.todense())

        res = self.op(scalar, ma1)
        assert_(isinstance(res, MapArray))
        assert_allclose(res.todense(),
                        self.op(scalar, ma1.todense()), atol=1e-15)

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

    def test_sparse_dense_interop(self):
        # dense + sparse densifies for scipy.sparse matrices.
        # So we try to be consistent here for sparse + dense or sparse += dense
        ma1 = self.ma.copy()
        dense = self.rhs.todense()

        res = self.op(ma1, dense)
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

    def test_dense_sparse_interop(self):
        # dense + sparse densifies for scipy.sparse matrices.
        # So we try to be consistent here for sparse + dense or sparse += dense
        ma1 = self.ma.copy()
        dense = self.rhs.todense()

        res = self.op(dense, ma1)
        assert_(isinstance(res, np.ndarray))
        assert_allclose(res,
                        self.op(dense, ma1.todense()), atol=1e-15)

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

    @skipif(not HAVE_SCIPY)
    def test_sparse_scipy_sparse_interop(self):
        # MapArray + csr_matrix should work
        ma1 = self.ma.copy()
        dense = self.rhs.todense()
        csr = sparse.csr_matrix(dense)

        res = self.op(ma1, csr)
        assert_(isinstance(res, MapArray))
        assert_allclose(res.todense(),
                        self.op(ma1.todense(), csr.toarray()), atol=1e-15)

        # also check the in-place version
        ma1 = self.ma.copy()
        dense = self.rhs.todense()
        csr = sparse.csr_matrix(dense)

        ma1 = self.iop(ma1, csr)
        assert_(isinstance(ma1, MapArray))
        assert_allclose(ma1.todense(),
                        self.op(self.ma.todense(), csr.toarray()), atol=1e-15)

    # multiplication fails :-(
    @skipif(not HAVE_SCIPY)
    def test_scipy_sparse_sparse_interop(self):

        if self.op == operator.mul:
            raise SkipTest("csr * map fails.")

        # csr_matrix + MapArray should work too
        ma1 = self.ma.copy()
        dense = self.rhs.todense()
        csr = sparse.csr_matrix(dense)

        res = self.op(csr, ma1)
        assert_(isinstance(res, MapArray))
        assert_allclose(res.todense(),
                        self.op(csr.toarray(), ma1.todense()), atol=1e-15)

        # also check the in-place version
        ma1 = self.ma.copy()
        dense = self.rhs.todense()
        csr = sparse.csr_matrix(dense)
        csr1 = csr.copy()

        csr = self.iop(csr, ma1)
        assert_(isinstance(csr, MapArray))
        assert_allclose(csr.todense(),
                        self.op(csr1.toarray(), self.ma.todense()), atol=1e-15)


############################ Addition

class TestArithmDouble(ArithmeticsMixin, TestCase):
    dtype = float


class TestArithmFloat(ArithmeticsMixin, TestCase):
    dtype = np.float32


class TestArithmPyInt(ArithmeticsMixin, TestCase):
    dtype = int


class TestArithmNpInt8(ArithmeticsMixin, TestCase):
    dtype = np.int8


class TestArithmNpInt16(ArithmeticsMixin, TestCase):
    dtype = np.int16


class TestArithmNpInt32(ArithmeticsMixin, TestCase):
    dtype = np.int32


class TestArithmNpInt64(ArithmeticsMixin, TestCase):
    dtype = np.int64


class TestArithmNpUInt8(ArithmeticsMixin, TestCase):
    dtype = np.uint8


class TestArithmNpUInt16(ArithmeticsMixin, TestCase):
    dtype = np.uint16


class TestArithmNpUInt32(ArithmeticsMixin, TestCase):
    dtype = np.uint32


class TestArithmNpUInt64(ArithmeticsMixin, TestCase):
    dtype = np.uint64


class TestArithmNpBool(ArithmeticsMixin, TestCase):
    dtype = np.bool


class TestArithmNpBool_(ArithmeticsMixin, TestCase):
    dtype = np.bool_


class TestArithmComplex(ArithmeticsMixin, TestCase):
    dtype = complex

############################ Multiplication

class MulMixin(ArithmeticsMixin):
    iop = operator.imul       # x = iop(x, y) is x *= y
    op = operator.mul


class TestMulDouble(MulMixin, TestCase):
    dtype = float


class TestMulFloat(MulMixin, TestCase):
    dtype = np.float32


class TestMulPyInt(MulMixin, TestCase):
    dtype = int


class TestMulNpInt8(MulMixin, TestCase):
    dtype = np.int8


class TestMulNpInt16(MulMixin, TestCase):
    dtype = np.int16


class TestMulNpInt32(MulMixin, TestCase):
    dtype = np.int32


class TestMulNpInt64(MulMixin, TestCase):
    dtype = np.int64


class TestMulNpUInt8(MulMixin, TestCase):
    dtype = np.uint8


class TestMulNpUInt16(MulMixin, TestCase):
    dtype = np.uint16


class TestMulNpUInt32(MulMixin, TestCase):
    dtype = np.uint32


class TestMulNpUInt64(MulMixin, TestCase):
    dtype = np.uint64


class TestMulPyBool(MulMixin, TestCase):
    dtype = bool


class TestMulNpBool(MulMixin, TestCase):
    dtype = np.bool


class TestMulNpBool_(MulMixin, TestCase):
    dtype = np.bool_


class TestMulComplex(MulMixin, TestCase):
    dtype = complex


############################ Subtraction

class SubMixin(ArithmeticsMixin):
    iop = operator.isub       # x = iop(x, y) is x -= y
    op = operator.sub


class TestSubDouble(SubMixin, TestCase):
    dtype = float


class TestSubFloat(SubMixin, TestCase):
    dtype = np.float32


class TestSubPyInt(SubMixin, TestCase):
    dtype = int


class TestSubNpInt32(SubMixin, TestCase):
    dtype = np.int32


class TestSubNpInt64(SubMixin, TestCase):
    dtype = np.int64


class TestSubNpUInt32(SubMixin, TestCase):
    dtype = np.uint32


class TestSubNpUInt64(SubMixin, TestCase):
    dtype = np.uint64


class TestSubComplex(SubMixin, TestCase):
    dtype = complex

#class TestSubPyBool(SubMixin, TestCase):
#    dtype = bool


############################ Type casting: (mostly) follow numpy

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


############################ Comparisons

class CmpMixin(object):

    def setUp(self):
        rndm = np.random.RandomState(1234)
        arr = rndm.random_sample(size=(3, 5)) * 10
        arr = arr.astype(self.ldtype)
        self.lhs = MapArray.from_dense(arr)

        arr1 = rndm.random_sample(size=(3, 5)) * 10 + 5
        arr1 = arr1.astype(self.rdtype)
        self.rhs = MapArray.from_dense(arr1)
        self.rhs.fill_value = -5

    def test_less(self):
        # array vs array
        assert_equal((self.lhs < self.rhs).todense(),
                     self.lhs.todense() < self.rhs.todense())

        # array vs scalar
        value = self.rhs.fill_value
        assert_equal((self.lhs < value).todense(),
                     self.lhs.todense() < value)

        # scalar vs array
        value = self.lhs.fill_value
        assert_equal((value < self.rhs).todense(),
                     value < self.rhs.todense())

        # array vs np.array (which densifies)
        assert_equal(self.lhs < self.rhs.todense(),
                     self.lhs.todense() < self.rhs.todense())

    @skipif(OLD_NUMPY, 'older numpy is too greedy w.r.t comparisons')
    def test_less_np_lhs(self):
        assert_equal(self.lhs.todense() < self.rhs,
                     self.lhs.todense() < self.rhs.todense())

    def test_leq(self):
        # array vs array
        assert_equal((self.lhs <= self.rhs).todense(),
                     self.lhs.todense() <= self.rhs.todense())

        # array vs scalar
        value = self.rhs.fill_value
        assert_equal((self.lhs <= value).todense(),
                     self.lhs.todense() <= value)

        # scalar vs array
        value = self.lhs.fill_value
        assert_equal((value <= self.rhs).todense(),
                     value <= self.rhs.todense())

        # array vs np.array (which densifies)
        assert_equal(self.lhs <= self.rhs.todense(),
                     self.lhs.todense() <= self.rhs.todense())

    @skipif(OLD_NUMPY, 'older numpy is too greedy w.r.t comparisons')
    def test_leq_np_lhs(self):
        assert_equal(self.lhs.todense() <= self.rhs,
                     self.lhs.todense() <= self.rhs.todense())

    def test_equal(self):
        # array vs array
        assert_equal((self.lhs == self.rhs).todense(),
                     self.lhs.todense() == self.rhs.todense())

        # array vs scalar
        value = self.rhs.fill_value
        assert_equal((self.lhs == value).todense(),
                     self.lhs.todense() == value)

        # scalar vs array
        value = self.lhs.fill_value
        assert_equal((value == self.rhs).todense(),
                     value == self.rhs.todense())

        # array vs np.array (which densifies)
        assert_equal(self.lhs == self.rhs.todense(),
                     self.lhs.todense() == self.rhs.todense())

    @skipif(OLD_NUMPY, 'older numpy is too greedy w.r.t comparisons')
    def test_equal_np_lhs(self):
        assert_equal(self.lhs.todense() == self.rhs,
                     self.lhs.todense() == self.rhs.todense())

    def test_neq(self):
        # array vs array
        assert_equal((self.lhs != self.rhs).todense(),
                     self.lhs.todense() != self.rhs.todense())

        # array vs scalar
        value = self.rhs.fill_value
        assert_equal((self.lhs != value).todense(),
                     self.lhs.todense() != value)

        # scalar vs array
        value = self.lhs.fill_value
        assert_equal((value != self.rhs).todense(),
                     value != self.rhs.todense())

        # array vs np.array (which densifies)
        assert_equal(self.lhs != self.rhs.todense(),
                     self.lhs.todense() != self.rhs.todense())

    @skipif(OLD_NUMPY, 'older numpy is too greedy w.r.t comparisons')
    def test_neq_np_lhs(self):
        assert_equal(self.lhs.todense() != self.rhs,
                     self.lhs.todense() != self.rhs.todense())

    def test_geq(self):
        # array vs array
        assert_equal((self.lhs >= self.rhs).todense(),
                     self.lhs.todense() >= self.rhs.todense())

        # array vs scalar
        value = self.rhs.fill_value
        assert_equal((self.lhs >= value).todense(),
                     self.lhs.todense() >= value)

        # scalar vs array
        value = self.lhs.fill_value
        assert_equal((value >= self.rhs).todense(),
                     value >= self.rhs.todense())

        # array vs np.array (which densifies)
        assert_equal(self.lhs >= self.rhs.todense(),
                     self.lhs.todense() >= self.rhs.todense())

    @skipif(OLD_NUMPY, 'older numpy is too greedy w.r.t comparisons')
    def test_geq_np_lhs(self):
        assert_equal(self.lhs.todense() >= self.rhs,
                     self.lhs.todense() >= self.rhs.todense())

    def test_greater(self):
        # array vs array
        assert_equal((self.lhs > self.rhs).todense(),
                     self.lhs.todense() > self.rhs.todense())

        # array vs scalar
        value = self.rhs.fill_value
        assert_equal((self.lhs > value).todense(),
                     self.lhs.todense() > value)

        # scalar vs array
        value = self.lhs.fill_value
        assert_equal((value > self.rhs).todense(),
                     value > self.rhs.todense())

        # array vs np.array (which densifies)
        assert_equal(self.lhs > self.rhs.todense(),
                     self.lhs.todense() > self.rhs.todense())

    @skipif(OLD_NUMPY, 'older numpy is too greedy w.r.t comparisons')
    def test_greater_np_lhs(self):
        # fails on numpy 1.6.2., works on 1.11.dev
        assert_equal(self.lhs.todense() > self.rhs,
                     self.lhs.todense() > self.rhs.todense())

    @skipif(not HAVE_SCIPY)
    def test_greater_coo_lhs(self):
        data, (row, col) = self.rhs.to_coo()
        coo = sparse.coo_matrix((data, (row, col)))
        assert_equal((self.lhs > coo).todense(),
                      self.lhs.todense() > coo.toarray())

    @skipif(True, 'coo > map fails')
    def test_greater_coo_lhs(self):
        data, (row, col) = self.rhs.to_coo()
        coo = sparse.coo_matrix((data, (row, col)))
        assert_equal((coo > self.lhs).todense(),
                      coo.toarray() > self.lhs.todense())


class CmpDoubleDouble(CmpMixin, TestCase):
    ldtype = float
    rdtype = float


class CmpDoubleInt(CmpMixin, TestCase):
    ldtype = float
    rdtype = int


############################ Matrix multiply

HAVE_MATMUL = hasattr(operator, 'matmul')


class MMulMixin(object):
    def setUp(self):
        rndm = np.random.RandomState(1234)
        arr = rndm.random_sample(size=(2, 3)) * 10
        arr = arr.astype(self.dtype_a)
        self.a = MapArray.from_dense(arr)

        arr1 = rndm.random_sample(size=(3, 2)) * 10
        arr1 = arr1.astype(self.dtype_b)
        self.b = MapArray.from_dense(arr1)

    @skipif(not HAVE_MATMUL)
    def test_basic(self):
        a, b = self.a, self.b
        x = eval("a @ b")
        assert_equal(x.dtype,
                     np.promote_types(a.dtype, b.dtype))
        assert_allclose(x.todense(),
                        np.dot(a.todense(), b.todense()), atol=1e-15)

    @skipif(not HAVE_MATMUL)
    def test_sparse_dense(self):
        # sparse @ dense -> dense
        a, b = self.a, self.b.todense()
        x = eval("a @ b")
        assert_(isinstance(x, np.ndarray))
        assert_equal(x.dtype,
                     np.promote_types(a.dtype, b.dtype))
        assert_allclose(x,
                        np.dot(a.todense(), b), atol=1e-15)

    @skipif(not HAVE_MATMUL)
    def test_dense_sparse(self):
        # dense @ sparse -> dense
        a, b = self.a.todense(), self.b
        x = eval("a @ b")
        assert_(isinstance(x, np.ndarray))
        assert_equal(x.dtype,
                     np.promote_types(a.dtype, b.dtype))
        assert_allclose(x,
                        np.dot(a, b.todense()), atol=1e-15)

    @skipif(not HAVE_MATMUL)
    def test_incompat_dims(self):
        a, b = self.a, self.b
        b[8, 12] = -101
        with assert_raises(ValueError):
            eval("a @ b")


class TestMMulFloat(MMulMixin, TestCase):
    dtype_a = float
    dtype_b = float


class TestMMulFloatInt(MMulMixin, TestCase):
    dtype_a = float
    dtype_b = int


class TestMMulIntInt(MMulMixin, TestCase):
    dtype_a = int
    dtype_b = int


############################ Indexing

def test_good_indexing():
    ma = MapArray()
    val = 2
    ma[2, 2] = val
    assert_equal(ma[2, 2], val)
    assert_equal(ma[-1, 2], val)
    assert_equal(ma[-2, 2], ma.fill_value)


def test_bad_indexing():
    # this is a nose test generator
    ma = MapArray()
    val = 2
    ma[2, 2] = val

    bad_indices = [
                   # numpy raises an IndexError for all of these:
                   (-4, 2),  # index is out of range
                   (1, 2, 3),  # too many dimensions
                   object(),   # integers. we needs them, my precious.
                   (object(),),
                   'enikibeniki',
                   ('boobabeeboob',),
                   (1j, 2),
                   1.5,      # floats shall not pass
                   (1.0, 2),

                   # some of these could work. Someday.
                   1,         # select a subarray
                   (1,),
                   (),        # what is it, a view
                   (slice(None,), 1),    # assorted slices
                   Ellipsis,
                   (Ellipsis,),
                   (1, Ellipsis),
                   [1, 2],            # fancy indexing
                   None,              # newaxis
                   (None, None),
                   (1, None),
                   (1, 2, None),
    ]
    for idx in bad_indices:
        yield check_bad_index_setitem, ma, idx

    # (0, 4) is out-of-bounds for getitem (setitem expands, so it's OK)
    bad_indices.append((0, 4))
    for idx in bad_indices:
        yield check_bad_index_getitem, ma, idx


def check_bad_index_getitem(ma, idx):
    with assert_raises(IndexError):
        ma[idx]


def check_bad_index_setitem(ma, idx):
    with assert_raises(IndexError):
        ma[idx] = 42


def test_fill_value_dtype():
    # test that the dtype defaults to the fill_value dtype

    # default is float
    m = MapArray()
    assert_equal(m.dtype, np.dtype(float)) 

    # if fill_value is provided, its dtype is used
    f = np.uint8(1)
    m = MapArray(fill_value=f)
    assert_equal(m.dtype, np.dtype(np.uint8))

    # ... unless it's given explicitly
    m = MapArray(fill_value=4.2, dtype=bool)
    assert_equal(m.dtype, np.dtype(bool))


if __name__ == "__main__":
    run_module_suite()
