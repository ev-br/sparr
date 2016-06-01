from cpython cimport PyObject
from libcpp.vector cimport vector

from common_types cimport single_index_type, slice_t
from fixed_cap cimport index_type

cdef extern from "slices.h" namespace "sparray":
    slice_t slice_from_pyslice(PyObject *sl_obj, Py_ssize_t len) except +
    slice_t slice_from_pyslice2(slice_t& sl, PyObject *pysl_obj) except +

    vector[slice_t] slices_from_pyslices(PyObject *pyslices,
                                         const index_type& shape) except +
    vector[slice_t] slices_from_pyslices2(const vector[slice_t]& sl,
                                          PyObject *pysl_obj) except +
