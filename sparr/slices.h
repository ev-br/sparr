#ifndef SPARR_SLICES
#define SPARR_SLICES
#include <Python.h>
#include <vector>
#include "common_types.h"
#include "fixed_cap.h"

namespace sparray {

/*
 * Convert a single python slice into a slice_t struct.
 */
slice_t 
slice_from_pyslice(PyObject *sl_obj, Py_ssize_t len);

/*
 * Combine slices for e.g. a[::2][::2]. The first slice is already a
 * slice_t struct, the second is a PyObject.
 */
slice_t
slice_from_pyslice2(const slice_t& sl, PyObject *pysl_obj);


/*
 * Convert an indexing tuple *of slices* into a slice_t vector.
 */
std::vector<slice_t>
slices_from_pyslices(PyObject *pyslices, const index_type& shape);


/*
 * Combine an indexing tuple of slices and a slice_t vector
 * for an n-D analog of a[::2][::2].
 */
std::vector<slice_t>
slices_from_pyslices2(const std::vector<slice_t>& sl, PyObject *pyslices);


} // namespace sparray

#endif
