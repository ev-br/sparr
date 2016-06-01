"""
Smoke test that combine_slices (which is, in fact, slice_from_pyslice2) 
is compatible with iterable[slice][slice].

"""
from __future__ import division, absolute_import, print_function
from numpy.testing import assert_equal

from sparr._slices import combine_slices

SLICES = [slice(None,), slice(None, None, 2), slice(None, None, 3),
          slice(None, None, -1), slice(None, None, -2), slice(None, None, -10),
          slice(1, None, 2), slice(2, None, -2),
          slice(1, 10, 2), slice(1, -2, 2),
          slice(-1, 1, -2), slice(-1, 1, 2)
]


def test_slices():
    for n in [5, 7, 24]:
        for slice1 in SLICES:
            for slice2 in SLICES:
                yield check_combine, slice1, slice2, n



def check_combine(slice1, slice2, n):
    lst = list(range(n))
    sl, length = combine_slices(slice1, slice2, n)

    res = lst[slice1][slice2]

    assert_equal(len(res), length)
    assert_equal(res,
                 [lst[_] for _ in range(*sl)])
