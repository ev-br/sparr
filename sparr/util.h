#ifndef SP_UTIL
#define SP_UTIL
#include <Python.h>
#include "numpy/ndarrayobject.h"

/*
 * basic utilities which require python.h or numpy includes, or both
 */

namespace sparray{

// adapted from numpy/core/src/multiarray/descriptor.c
inline int
_is_tuple_of_integers(PyObject *obj, int len=2)
{
    if (!PyTuple_Check(obj)) {
        return 0;
    }
    if (PyTuple_GET_SIZE(obj) != len) {
        return 0;
    }
    for (int i = 0; i < PyTuple_GET_SIZE(obj); i++) {
        if (!PyArray_IsIntegerScalar(PyTuple_GET_ITEM(obj, i))) {
            return 0;
        }
    }
    return 1;
}


} // end namespace sparray

#endif
