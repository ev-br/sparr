"""
Provide the python exposure of slice_from_pyslice2.
For tests only. This is not public.
"""
from cpython.ref cimport PyObject
from sparr.slices cimport slice_t, slice_from_pyslice, slice_from_pyslice2

def combine_slices(slice1, slice2, len):
    cdef slice_t c_sl = slice_from_pyslice(<PyObject *>slice1, len)
    cdef slice_t c_sl2 = slice_from_pyslice2(c_sl, <PyObject *>slice2)
    return (c_sl2.start, c_sl2.stop, c_sl2.step), c_sl2.slicelength
