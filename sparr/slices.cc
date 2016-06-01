#include <iostream>
#include <stdexcept>
#include <vector>
#include "slices.h"
#include "fixed_cap.h"


sparray::slice_t 
sparray::slice_from_pyslice(PyObject *sl_obj, Py_ssize_t len) {

    if (!sl_obj)
        throw std::runtime_error("sl_obj is NULL");

    if (!PySlice_Check(sl_obj))
        throw std::runtime_error("sl_obj fails PySlice_Check");   

#if ((PY_MAJOR_VERSION >= 3) && (PY_MINOR_VERSION >=2))
    PyObject *obj = sl_obj;
#else
    PySliceObject *obj = (PySliceObject *)sl_obj;
#endif
    int flag;
    sparray::slice_t res;
    flag = PySlice_GetIndicesEx(obj, len,
                                &res.start, &res.stop, &res.step, &res.slicelength);
    if (flag)
        throw std::runtime_error("GetIndicesEx failed");

    return res;
}


sparray::slice_t
sparray::slice_from_pyslice2(const sparray::slice_t& sl, PyObject *pysl_obj) {

    if (!pysl_obj)
        throw std::runtime_error("pysl_obj is NULL");

    if (!PySlice_Check(pysl_obj))
        throw std::runtime_error("pysl_obj fails PySlice_Check");   

#if ((PY_MAJOR_VERSION >= 3) && (PY_MINOR_VERSION >=2))
    PyObject *obj = pysl_obj;
#else
    PySliceObject *obj = (PySliceObject *)pysl_obj;
#endif
    int flag;
    sparray::slice_t sl2;
    flag = PySlice_GetIndicesEx(obj, sl.slicelength,
                                &sl2.start, &sl2.stop, &sl2.step, &sl2.slicelength);
    if (flag)
        throw std::runtime_error("GetIndicesEx failed");
        
    sparray::slice_t res;
    res.start = sl2.start*sl.step + sl.start;
    res.step = sl2.step*sl.step;
    res.slicelength = sl2.slicelength;
    res.stop = res.start + res.step*res.slicelength;

    return res;
}


std::vector<sparray::slice_t>
sparray::slices_from_pyslices(PyObject *pyslices, const sparray::index_type& shape) {

    if (!pyslices)
        throw std::runtime_error("pyslices is NULL");

    if (!PyTuple_Check(pyslices))
        throw std::runtime_error("pyslices fails PyTuple_Check");

    int pyslice_length = PyTuple_GET_SIZE(pyslices);

    if (pyslice_length != shape.ndim())
        throw std::runtime_error("pyslices: dimension mismatch.");

    std::vector<sparray::slice_t> res;

    for(int i=0; i < pyslice_length; i++) {
        PyObject *pysl = PyTuple_GetItem(pyslices, i);
        if (!pysl)
            throw std::runtime_error("Out of bounds");

        sparray::slice_t s = sparray::slice_from_pyslice(pysl, (Py_ssize_t)shape[i]);
        res.push_back(s);
    }

    return res;
}


std::vector<sparray::slice_t>
sparray::slices_from_pyslices2(const std::vector<slice_t>& sl, PyObject *pyslices) {

    if (!pyslices)
        throw std::runtime_error("pyslices is NULL");

    if (!PyTuple_Check(pyslices))
        throw std::runtime_error("pyslices fails PyTuple_Check");

    int pyslice_length = PyTuple_GET_SIZE(pyslices);

    if (pyslice_length != (int)sl.size())
        throw std::runtime_error("pyslices: dimension mismatch.");

    std::vector<sparray::slice_t> res;
    for(int i=0; i < pyslice_length; i++) {
        PyObject *pysl = PyTuple_GetItem(pyslices, i);
        if (!pysl)
            throw std::runtime_error("Out of bounds");

        sparray::slice_t s = sparray::slice_from_pyslice2(sl[i], pysl);
        res.push_back(s);
    }

    return res;
}

