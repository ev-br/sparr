from __future__ import division, print_function, absolute_import

import operator
from sys import getrefcount as refc

import numpy as np
from numpy.testing import (run_module_suite, TestCase, assert_equal, assert_,
                           assert_allclose, assert_raises,)
from numpy.testing.decorators import knownfailureif, skipif

from .. import MapArray

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

