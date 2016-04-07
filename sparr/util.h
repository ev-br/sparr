#ifndef SP_UTIL
#define SP_UTIL
#include <Python.h>
#include "numpy/ndarrayobject.h"
#include "fixed_cap.h"
#include "sp_map.h"


/*
 * basic utilities which require python.h or numpy includes, or both
 */

namespace sparray{

/*
 * Try converting a tuple of integers into an index_type struct.
 *
 * adapted from numpy/core/src/multiarray/descriptor.c
 */
inline int
_is_tuple_of_integers(PyObject *obj, index_type& idx, int len)
{
    PyObject *x;

    if (!PyTuple_Check(obj)) {
        return 0;
    }
    if (PyTuple_GET_SIZE(obj) != len) {
        return 0;
    }
    for (int i = 0; i < PyTuple_GET_SIZE(obj); i++) {
        x = PyTuple_GET_ITEM(obj, i); 
        if (!PyArray_IsIntegerScalar(x)) {
            return 0;
        }
        idx[i] = PyInt_AsLong(x);
    }
    return 1;
}


/*
 * Check if `idx` is a valid index to the array `*arr`, taking into account negative indices.
 * For negative indices, modify the `idx` argument in-place.
 * For invalid indices, raise IndexError.
 * Depending on the value of `is_shape_fixed`, decide if expanding the array
 * is allowed or not.
 *
 * The dimensionality of `idx` and `arr` is NOT checked and it must be validated
 * beforehand.
 */
template<typename T>
inline PyObject*
validate_index(index_type& idx, const map_array_t<T> *arr, int is_shape_fixed)
{
    index_type shp = arr->shape();
    for(size_t j = 0; j < arr->ndim(); ++j){
        if (idx[j] < 0){
            idx[j] += shp[j];
        }

        // check if still out of bounds
        if ((idx[j] < 0) ||
            (is_shape_fixed && (idx[j] >= shp[j]))){
                PyErr_Format(PyExc_IndexError, "index %ld is out of bounds for "
                            "axis %zu with size %ld", idx[j], j, shp[j]);
                return NULL;
        }
    }

    // return something non-NULL to signal no error
    Py_RETURN_TRUE;
}


} // end namespace sparray

#endif
