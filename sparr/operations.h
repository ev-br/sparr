#ifndef SP_OPERATIONS_H
#define SP_OPERATIONS_H
#include <iostream>
#include <stdexcept>

#include "views.h"

namespace sparray {


/*
 * Convert to a dense representation (C order).
 * Caller's responsible for allocating memory.
 */
template<typename T, typename I>
inline void
todense(abstract_view_t<T, I> *src,
        void* dest,
        const I num_elem)
{
    typedef abstract_view_t<T, I> Array;

    if (!src){ throw std::invalid_argument("todense: src is NULL."); }
    if (num_elem < 0){ throw std::runtime_error("num_elem < 0"); }
    if (num_elem == 0){ return; }

    // fill the background
    T *_dest = static_cast<T*>(dest);
    std::fill(_dest, _dest + num_elem, src->fill_value());

    // fill nonzero elements
    typename Array::const_iterator it = src->cbegin();
    for(; it != src->cend(); ++it){
        I idx = src->_flat_index(it->first);
        assert(idx < num_elem);
        _dest[idx] = it->second;
    }
}


/*
 * Convert to a COO triplet of (data, row, column) arrays.
 * Caller's responsible for allocating memory.
 */
template<typename T, typename I>
inline void
to_coo(const abstract_view_t<T, I>  *src,
       void *data, void *stacked_indices,
       const I num_elem)
{
    typedef abstract_view_t<T, I> Array;

    if (!src)
        throw std::invalid_argument("to_coo: src is NULL.");

    if (!data)
        throw std::invalid_argument("to_coo: data is NULL.");
    T *_data = static_cast<T*>(data);

    if (!stacked_indices)
        throw std::invalid_argument("to_coo: indices is NULL.");
    I *_indices = static_cast<I*>(stacked_indices);

    if (num_elem < 0)
        throw std::runtime_error("num_elem < 0");
    if (num_elem == 0){ return; }

    typename Array::const_iterator it = src->cbegin();
    I j = 0;
    for(; it != src->cend(); ++it){
        assert(j < num_elem);
        _data[j] = it->second;

        typename Array::index_type idx = it->first;
        for(int d=0; d < src->ndim(); ++d) {
            _indices[j + d*num_elem] = idx[d];
        }
        j += 1;
    }
}


/*
 * x <- alpha f(x) + beta x
 */
template<typename T, typename I>
inline void
inplace_unary(abstract_view_t<T, I> *self,
              T (*op)(T x),
              T alpha=1, T beta=0)
{
    typedef abstract_view_t<T, I> Array;

    if(!self)
        throw std::invalid_argument("unary_op: self is NULL");

    typename Array::iterator it = self->begin();
    for (; it != self->end(); ++it){
        self->set_one(it, alpha * (*op)(it->second) + beta * it->second);
    }
    T fill_value = alpha* (*op)(self->fill_value()) + beta * self->fill_value();
    self->set_fill_value(fill_value);
}


/*
 * x <- alpha f(x, y) + beta x, \foreach x in self, \foreach y in other
 */
template<typename T, typename I>
inline void
inplace_binop(T (*binop)(T x, T y),
              abstract_view_t<T, I> *self,
              const abstract_view_t<T, I> *other,
              T alpha=1, T beta=0)
{
    typedef abstract_view_t<T, I> Array;

    if(!self)
        throw std::invalid_argument("binop: self is NULL");
    if(!other)
        throw std::invalid_argument("binop: other is NULL");

    // check that the dimensions are compatible
    for(int j=0; j<self->ndim(); ++j){
        if(self->shape()[j] != other->shape()[j]){
            throw std::invalid_argument("Binop: incompatible dimensions.");
        }
    }

    // run over the nonzero elements of *this. This is O(n1 * log(n2))
    typename Array::iterator it = self->begin();
    typename Array::const_iterator it_other;
    typename Array::index_type idx;
    for(; it != self->end(); ++it){
        idx = it->first;
        it_other = other->find(idx);
        T y = (it_other != other->cend()) ? it_other->second 
                                          : other->fill_value();
        self->set_one(it, alpha * (*binop)(it->second, y) + beta * it->second);
    }

    if(other != self){
        // run over the nonzero elements of *other; those which are present in both
        // *this and *other have been taken care of already. Insert new ones
        // into *this. This loop's complexity is O(n2 * log(n1))
        typename Array::index_type idx;
        typename Array::const_iterator it_other = other->cbegin();
        for(; it_other != other->cend(); ++it_other){
            idx = it_other->first;
            T value = alpha * (*binop)(self->fill_value(), it_other->second) +
                      beta * self->fill_value();
            self->_insert(std::make_pair(idx, value));
        }
    }

    // update fill_value
    T fill_value = alpha * (*binop)(self->fill_value(), other->fill_value()) +
                   beta * self->fill_value();
    self->set_fill_value(fill_value);
}


/*
 * z <- alpha f(x, y) + beta x, \foreach x in arg1, \foreach y in arg2
 */
template<typename S, typename T, typename I>
inline void
apply_binop(T (*binop)(S x, S y),
            abstract_view_t<T, I> *self,
            const abstract_view_t<S, I> *arg1,
            const abstract_view_t<S, I> *arg2)
{
    typedef abstract_view_t<T, I> ArrayT;
    typedef abstract_view_t<S, I> ArrayS;

    // XXX maybe generalize to $alpha f(x, y) + beta y$
    if(!self)
        throw std::invalid_argument("apply_binop: self is NULL");
    if(!arg1)
        throw std::invalid_argument("apply_binop: arg1 is NULL");
    if(!arg2)
        throw std::invalid_argument("apply_binop: arg2 is NULL");

    // check that the dimensions are compatible
    for(int j=0; j < self->ndim(); j++){
        if(arg1->shape()[j] != arg2->shape()[j]){
            throw std::invalid_argument("Binop: incompatible dimensions.");
        }
    }

    // result shape is known, set it right away
    self->set_shape(arg1->shape());

    // run over the nonzero elements of *arg1. This is O(n1 * log(n2) * log(n1))
    std::pair<typename ArrayT::iterator, bool> p;
    typename ArrayT::index_type idx;
    typename ArrayS::const_iterator it, it2;
    for(it = arg1->cbegin(); it != arg1->cend(); ++it){
        idx = it->first;
        it2 = arg2->find(idx);
        S y = (it2 != arg2->cend()) ? it2->second
                                    : arg2->fill_value();

        T value = (*binop)(it->second, y);

        p = self->_insert(std::make_pair(idx, value));
        if(!(p.second)){ p.first->second = value; }   // overwrite what was there at index idx
    }

    if(arg2 != arg1){
        // run over the nonzero elements of *arg2; those which are present in both
        // *arg1 and *arg2 have been taken care of already.
        // This loop's complexity is O(n2 * log(n1) * log(n2))
        std::pair<typename ArrayT::iterator, bool> p;
        typename ArrayT::index_type idx;
        typename ArrayS::const_iterator it1, it2;
        for(it2 = arg2->cbegin(); it2 != arg2->cend(); ++it2) {
            idx = it2->first;
            it1 = arg1->find(idx);
            if (it1 == arg1->cend()) {
                // it1->second is present in arg2 but not in arg1
                T value = (*binop)(arg1->fill_value(), it2->second);

                p = self->_insert(std::make_pair(idx, value));
                if(!(p.second)){ p.first->second = value; }

            }
        }
    }

    // update fill_value
    T fill_value = (*binop)(arg1->fill_value(), arg2->fill_value());
    self->set_fill_value(fill_value);
}


/*
 * Matrix multiplication: C <- A @ B
 */
template<typename T, typename I>
inline void
apply_mmul(abstract_view_t<T, I> *C,
           const abstract_view_t<T, I> *A,
           const abstract_view_t<T, I> *B)
{
    typedef abstract_view_t<T, I> Array;

    if(!C)
        throw std::invalid_argument("mmul: C is NULL");
    if(!A)
        throw std::invalid_argument("mmul: A is NULL");
    if(!B)
        throw std::invalid_argument("mmul: B is NULL");

    assert(C->ndim() == 2);
    assert(A->ndim() == 2);
    assert(B->ndim() == 2);

    if(A->shape()[1] != B->shape()[0])
        throw std::invalid_argument("MMul: incompatible dimensions.");

    // the shape of the result is known, set it right away
    index_type shp(C->ndim());
    shp[0] = A->shape()[0];
    shp[1] = B->shape()[1];
//    I arr[2];
//    arr[0] = A->shape()[0];
//    arr[1] = B->shape()[1];
    C->set_shape(shp);

    // XXX non-zero fill_values are not implemented. The result is likely
    // dense anyway, so having to convert to dense explicitly is not *that* bad.
    if((A->fill_value() != 0) || (B->fill_value() != 0))
        throw std::runtime_error("Non-zero fill_values not handled yet.");

    // C_{ik} = alpha A_{ij} B_{jk} + beta C_{ik}
    std::pair<typename Array::iterator, bool> p;
    typename Array::iterator itC;
    typename Array::const_iterator itA, itB;
    typename Array::index_type idxA, idxB;

    T Aij, Bjk, Cik, value;
    for (itA = A->cbegin(); itA != A->cend(); ++itA){
        Aij = itA->second;
        idxA = itA->first;
        shp[0] = idxA[0];
        for(itB = B->cbegin(); itB != B->cend(); ++itB){
            idxB = itB->first;
            if( idxA[1] != idxB[0])
                continue;                       // XXX row/column_iterators?

            Bjk = itB->second;
            shp[1] = idxB[1];
            typename Array::index_type idx(shp);
            itC = C->find(idx);
            Cik = (itC != C->end()) ? itC->second
                                    : C->fill_value();
            value = Aij*Bjk + Cik;

            p = C->_insert(std::make_pair(idx, value));
            if(!(p.second)){ C->set_one(p.first, value); }
        }
    }
}


} // namespace sparray


#endif
