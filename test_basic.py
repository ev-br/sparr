import operator

import numpy as np
from numpy.testing import (run_module_suite, TestCase, assert_equal, assert_,
                           assert_allclose, assert_raises,)
from numpy.testing.decorators import knownfailureif, skipif

from sp_map import MapArray


class TestBasic(TestCase):
    def test_ctor(self):
        ma = MapArray()
        assert_equal(ma.ndim, 2)
        assert_equal(ma.shape, (0, 0))
        assert_allclose(ma.fill_value, 0., atol=1e-15)

    def test_fill_value_mutable(self):
        ma = MapArray()
        ma.fill_value = -1.
        assert_equal(ma.fill_value, -1)
        with assert_raises(TypeError):
            ma.fill_value = 'lalala'

    def test_basic_insert(self):
        ma = MapArray()
        ma[1, 1] = -1
        assert_equal(ma.shape, (2, 2))
        assert_equal(ma.ndim, 2)
        assert_equal(ma.count_nonzero(), 1)
        assert_allclose(ma.todense(),
                        np.array([[0, 0],
                                  [0, -1]]), atol=1e-15)

        ma[2, 3] = 8
        assert_equal(ma.ndim, 2)
        assert_equal(ma.shape, (3, 4))
        assert_equal(ma.count_nonzero(), 2)
        assert_allclose(ma.todense(),
                        np.array([[0, 0, 0, 0],
                                  [0, -1, 0, 0],
                                  [0, 0, 0, 8]]), atol=1e-15)

    def test_todense_fillvalue(self):
        ma = MapArray()
        ma[2, 3] = 8
        ma[1, 1] = -1
        ma.fill_value = 1
        assert_equal(ma.count_nonzero(), 2)
        assert_allclose(ma.todense(),
                        np.array([[1, 1, 1, 1],
                                  [1, -1, 1, 1],
                                  [1, 1, 1, 8]]), atol=1e-15)

    @skipif(True, '32 bit indices')
    def test_int_overflow(self):
        # check array size s.t. a flat index overflows the C int range
        ma = MapArray()
        j = np.iinfo(np.int32).max + 1
        ma[1, j] = 1.
        ma.todense()

    def test_copy(self):
        ma = MapArray()
        ma[1, 1] = 1.

        # operate on a copy; the original must be intact
        ma1 = ma.copy()
        ma1[2, 4] = 3.

        assert_equal(ma1.shape, (3, 5))
        assert_allclose(ma1.todense(),
                        np.array([[0, 0, 0, 0, 0],
                                  [0, 1, 0, 0, 0], 
                                  [0, 0, 0, 0, 3]]), atol=1e-15)
        assert_equal(ma.shape, (2, 2))
        assert_allclose(ma.todense(),
                        np.array([[0, 0],
                                  [0, 1]]), atol=1e-15)


class TestArithmetics(TestCase):
    ma = MapArray()
    ma[1, 1] = 1.
    ma[2, 4] = 2.

    rhs = MapArray()
    rhs[2, 4] = 3.
    rhs.fill_value = -8

    iop = operator.iadd
    op = operator.add

    def test_inplace_add_scalar(self):
        ma1 = self.ma.copy()
        self.iop(ma1, 4)      # IOW, ma1 += 4

        assert_equal(ma1.shape, self.ma.shape)
        assert_allclose(ma1.todense(),
                        self.op(self.ma.todense(), 4.), atol=1e-15)

    def test_inplace_add_unsupported_type_obj(self):
        ma1 = self.ma.copy()
        with assert_raises(TypeError):
            self.iop(ma1, 'lalala')

        with assert_raises(TypeError):
            self.iop(ma1, None)

        with assert_raises(TypeError):
            self.iop(ma1, self.rhs.todense())

    def test_inplace_add_sparse(self):
        ma1 = self.ma.copy()
        rhs = self.rhs.copy()
        self.iop(ma1, rhs)

        # the LHS is operated on, and RHS is intact
        assert_allclose(ma1.todense(),
                        self.ma.todense() + self.rhs.todense(), atol=1e-15)
        assert_allclose(rhs.todense(), self.rhs.todense(), atol=1e-15)

    def test_inplace_add_wrong_shape(self):
        # incompatible shapes should raise ValueErrors
        ma1 = self.ma.copy()
        rhs = self.rhs.copy()
        rhs[8, 9] = -101
        with assert_raises(ValueError):
            self.iop(ma1, rhs)



if __name__ == "__main__":
    run_module_suite()
