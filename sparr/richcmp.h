#ifndef SP_BOOLEANS_H
#define SP_BOOLEANS_H
#include <Python.h>

#include "elementwise_ops.h"
#include "operations.h"

namespace sparray {

/*
 * Compute the array of `*lhs < *rhs` et al elementwise, store the result in *trg.
 * Caller's responsible for allocating memory
 */
template<typename S, typename T, typename I>
inline void
rich_compare(abstract_view_t<T, I> *trg,
             const abstract_view_t<S, I> *lhs,
             const abstract_view_t<S, I> *rhs,
             int opid)
{
    switch(opid) {
    case Py_LT:
        apply_binop<S, T, npy_intp>(less<S>, trg, lhs, rhs, 1);
        break;
    case Py_LE:
        apply_binop<S, T, npy_intp>(less_equal<S>, trg, lhs, rhs, 1);
        break;
    case Py_EQ:
        apply_binop<S, T, npy_intp>(equal<S>, trg, lhs, rhs, 1);
        break;
    case Py_NE:
        apply_binop<S, T, npy_intp>(not_equal<S>, trg, lhs, rhs, 1);
        break;
    case Py_GT:
        apply_binop<S, T, npy_intp>(greater_equal<S>, trg, lhs, rhs, 1);
        break;
    case Py_GE:
        apply_binop<S, T, npy_intp>(greater_equal<S>, trg, lhs, rhs, 1);
        break;
    default:
        throw std::runtime_error("Unknown operation.");
    }
}

} // namespace sparray

#endif
