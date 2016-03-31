#ifndef SP_UTIL
#define SP_UTIL
#include <Python.h>
#include "numpy/ndarrayobject.h"
#include "fixed_cap.h"


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
_is_tuple_of_integers(PyObject *obj, index_type& idx)
{
    PyObject *x;

    if (!PyTuple_Check(obj)) {
        return 0;
    }
    if (PyTuple_GET_SIZE(obj) != idx.ndim()) {
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


} // end namespace sparray

#endif
