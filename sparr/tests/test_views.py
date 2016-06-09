from __future__ import division, print_function, absolute_import

import operator
from sys import getrefcount as refc

import numpy as np
from numpy.testing import (run_module_suite, TestCase, assert_equal, assert_,
                           assert_allclose, assert_raises,)
from numpy.testing.decorators import skipif, knownfailureif as knf

from .. import MapArray

from .test_slices import SLICES


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
    r0 = refc(m)

    mm = m[...]
    mm[0, 1] = 3.

    assert_allclose(m.todense(), np.array([[0, 3], [0, 2]]), atol=1e-15)

    del m
    assert_allclose(mm.todense(), np.array([[0, 3], [0, 2]]), atol=1e-15)


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


def test_two_ellipsis():
    # a[..., 2, ...] is deprecated in numpy; we do not allow it either
    a = MapArray(shape=(2, 3, 4))
    with assert_raises(IndexError):
        a[..., 1, ...]
