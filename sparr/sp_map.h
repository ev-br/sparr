#ifndef SP_MAP_H
#define SP_MAP_H
#include<map>
#include<stdexcept>
#include"fixed_cap.h"


namespace sparray {


template<typename T, typename I=single_index_type>
struct map_array_t
{
    typedef std::map<fixed_capacity<I>,
                      T,
                      fixed_capacity_cmp<I> > map_type;
    typedef typename map_type::value_type value_type;    // pair<key, value>
    typedef typename map_type::key_type index_type;
    typedef typename map_type::const_iterator const_iterator;
    typedef typename map_type::iterator iterator;

    map_array_t(const size_t num_dim = 2, const T& fill_value = 0);
    map_array_t(const map_array_t& other);
    template<typename S> void copy_from(const map_array_t<S, I> *src);

    size_t ndim() const { return m_ndim; }
    index_type shape() const { return shape_; }
    void set_shape(const index_type& idx);

    T fill_value() const { return fill_value_; }
    void set_fill_value(const T& value) { fill_value_ = value; }

    size_t count_nonzero() const { return data_.size(); }

    const_iterator begin() const { return data_.begin(); }
    const_iterator end() const { return data_.end(); }

    iterator begin() { return data_.begin(); }
    iterator end() { return data_.end(); }

    const_iterator find(const index_type& idx) const { return data_.find(idx);}
    iterator find(const index_type& idx) { return data_.find(idx);}

    std::pair<iterator, bool> insert(const value_type& val){ return data_.insert(val); }
    iterator insert(iterator hint, const value_type& val) { return data_.insert(hint, val); }

    // retrieve / set a single element
    T get_one(const index_type& idx) const;
    void set_one(const index_type& idx, const T& value);

    // indexing helpers
    I _flat_index(const index_type& index) const;
 
    private:
        map_type data_;
        index_type shape_;
        T fill_value_;
        int m_ndim;
};


/*
 * Operations on the map_array_t objects.
 * 
 * NB: the struct itself is a kludge for working around the fact that
 *     default template arguments are not allowed in function templates.
 *     It should probably go away once the I and num_dim template arguments
 *     are sorted out.
 */
template<typename T, typename I=single_index_type>
struct operations
{
    // convert to a dense representation (C order). Caller's responsible for
    // allocating memory.
    void todense(const map_array_t<T, I>* src,
                 void* dest,
                 const I num_elem);

    // convert to a COO triplet of (data, row, column) arrays.
    // Caller's responsible for allocating memory.
    void to_coo(const map_array_t<T, I>* src,
                void* data, void* row, void* clmn,
                const I num_elem);

    // x <- alpha f(x) + beta x
    void inplace_unary(map_array_t<T, I> *self,
                       T (*op)(T x),
                       const T alpha=1, const T beta=0);

    // x <- alpha f(x, y) + beta x, \foreach x in self, \foreach y in other
    void inplace_binop(T (*binop)(T x, T y),
                       map_array_t<T, I> *self, 
                       const map_array_t<T, I>* other,
                       const T alpha=1, const T beta=0);

    // z <- alpha f(x, y) + beta x, \foreach x in arg1, \foreach y in arg2
    template<typename S> void apply_binop(T (*binop)(S x, S y),
                                          map_array_t<T, I> *self,
                                          const map_array_t<S, I> *arg1,
                                          const map_array_t<S, I> *arg2);

    // Matrix multiplication: C <- A @ B
    void apply_mmul(map_array_t<T, I> *C,
                    const map_array_t<T, I>* A,
                    const map_array_t<T, I>* B);

};


/////////////////// IMPLEMENTATIONS of map_array_t methods.

template<typename T, typename I>
inline map_array_t<T, I>::map_array_t(size_t num_dim, const T& fill_value)
{
    for(size_t j=0; j < num_dim; ++j){
        shape_[j] = 0;
    }
    fill_value_ = fill_value;
    m_ndim = num_dim;
}


template<typename T, typename I>
inline map_array_t<T, I>::map_array_t(const map_array_t<T, I>& other)
{
    copy_from(&other);
}


template<typename T, typename I>
template<typename S>
inline void
map_array_t<T, I>::copy_from(const map_array_t<S, I> *src)
{
    if(!src)
        throw std::invalid_argument("copy from NULL");

    if(this->ndim() != src->ndim())
        throw std::runtime_error("Dimensions mismatch.");

    typename map_array_t<S, I>::const_iterator it = src->begin();
    typename map_array_t<T, I>::iterator hint = this->begin();

    for(; it != src->end(); ++it){
        T value = static_cast<T>(it->second);
        hint = this->insert(hint, std::make_pair(it->first, value));
        hint->second = value;
    }

    for(size_t j=0; j < ndim(); ++j){
        shape_[j] = src->shape()[j];
    }

    fill_value_ = static_cast<T>(src->fill_value());
}


template<typename T, typename I>
inline T
map_array_t<T, I>::get_one(const map_array_t::index_type& idx) const
{
    const_iterator it = this->find(idx);
    return (it != this->end()) ? it->second : fill_value_;
}


template<typename T, typename I>
inline void
map_array_t<T, I>::set_one(const map_array_t::index_type& idx, const T& value)
{
    // XXX Can use a hint if inserting consequtive values.
    std::pair<iterator, bool> p = this->insert(std::make_pair(idx, value));
    p.first->second = value;

    // update the shape if needed
    for(size_t j=0; j < ndim(); ++j){
        if(idx[j] >= shape_[j]){
            shape_[j] = idx[j] + 1;
        }
    }
}


template<typename T, typename I>
inline void 
map_array_t<T, I>::set_shape(const index_type& shp)
{
    index_type min_shape = get_min_shape(*this);

    for(size_t j = 0; j < ndim(); ++j){
        if(shp[j] >= min_shape[j]){
            shape_[j] = shp[j];
        }
        else{
            throw std::domain_error("set_shape: arg too small!");
        }
    }
}


/* Flat index to an array. C order, 2D only.
 */
template<typename T, typename I>
inline I
map_array_t<T, I>::_flat_index(const typename map_array_t<T, I>::index_type& index) const
{
    assert(ndim() == 2);
    I stride = shape_[1];
    return index[0]*stride + index[1];
}



///////////////////////////////////////// IMPL operations

template<typename T, typename I>
inline void
operations<T, I>::todense(const map_array_t<T, I>* src,
                                   void* dest,
                                   const I num_elem)
{
    if (!src){ throw std::invalid_argument("todense: src is NULL."); }
    if (num_elem < 0){ throw std::runtime_error("num_elem < 0"); }
    if (num_elem == 0){ return; }

    // fill the background
    T *_dest = static_cast<T*>(dest);
    std::fill(_dest, _dest + num_elem, src->fill_value());

    // fill nonzero elements
    typename map_array_t<T, I>::const_iterator it = src->begin();
    for(; it != src->end(); ++it){
        I idx = src->_flat_index(it->first);
        assert(idx < num_elem);
        _dest[idx] = it->second;
    }
}


template<typename T, typename I>
inline void
operations<T, I>::to_coo(const map_array_t<T, I>* src,
                         void* data, void* row, void* clmn,
                         const I num_elem)
{
    assert(src->ndim() == 2);

    if (!src){ throw std::invalid_argument("to_coo: src is NULL."); }

    if (!data)
        throw std::invalid_argument("to_coo: data is NULL.");
    T *_data = static_cast<T*>(data);

    if (!row)
        throw std::invalid_argument("to_coo: row is NULL.");
    I *_row = static_cast<I*>(row);

    if (!clmn)
        throw std::invalid_argument("to_coo: clmn is NULL.");
    I *_clmn = static_cast<I*>(clmn);

    if (num_elem < 0)
        throw std::runtime_error("num_elem < 0");
    if (num_elem == 0){ return; }

    typename map_array_t<T, I>::index_type idx;
    typename map_array_t<T, I>::const_iterator it = src->begin();
    I j = 0;
    for(; it != src->end(); ++it){
        assert(j < num_elem);
        idx = it->first;
        _data[j] = it->second;
        _row[j] = idx[0];
        _clmn[j] = idx[1];
        j += 1;
    }
}


template<typename T, typename I>
inline void
operations<T, I>::inplace_unary(map_array_t<T, I> *self,
                                         T (*op)(T x),
                                         T alpha, T beta)
{
    if(!self)
        throw std::invalid_argument("unary_op: self is NULL");

    typename map_array_t<T, I>::iterator it = self->begin();
    for (; it != self->end(); ++it){
        it->second = alpha * (*op)(it->second) + beta * it->second;
    }
    T fill_value = alpha* (*op)(self->fill_value()) + beta * self->fill_value();
    self->set_fill_value(fill_value);
}


template<typename T, typename I>
inline void
operations<T, I>::inplace_binop(T (*binop)(T x, T y),
                                         map_array_t<T, I> *self,
                                         const map_array_t<T, I> *other,
                                         T alpha, T beta)
{
    if(!self)
        throw std::invalid_argument("binop: self is NULL");
    if(!other)
        throw std::invalid_argument("binop: other is NULL");

    // check that the dimensions are compatible
    for(size_t j=0; j<self->ndim(); ++j){
        if(self->shape()[j] != other->shape()[j]){
            // TODO: probably want to output the dimensions here
            throw std::invalid_argument("Binop: incompatible dimensions.");
        }
    }

    // run over the nonzero elements of *this. This is O(n1 * log(n2))
    typename map_array_t<T, I>::iterator it = self->begin();
    typename map_array_t<T, I>::const_iterator it_other;
    typename map_array_t<T, I>::index_type idx;
    for(; it != self->end(); ++it){
        idx = it->first;
        it_other = other->find(idx);
        T y = (it_other != other->end()) ? it_other->second 
                                         : other->fill_value();
        it->second = alpha * (*binop)(it->second, y) + beta * it->second;
    }

    if(other != self){
        // run over the nonzero elements of *other; those which are present in both
        // *this and *other have been taken care of already. Insert new ones
        // into *this. This loop's complexity is O(n2 * log(n1))
        typename map_array_t<T, I>::index_type idx;
        typename map_array_t<T, I>::const_iterator it_other = other->begin();
        for(; it_other != other->end(); ++it_other){
            idx = it_other->first;
            T value = alpha * (*binop)(self->fill_value(), it_other->second) +
                      beta * self->fill_value();
            self->insert(std::make_pair(idx, value));
        }
    }

    // update fill_value
    T fill_value = alpha * (*binop)(self->fill_value(), other->fill_value()) +
                   beta * self->fill_value();
    self->set_fill_value(fill_value);
}


template<typename T, typename I>
template<typename S>
inline void
operations<T, I>::apply_binop(T (*binop)(S x, S y),
                                       map_array_t<T, I> *self,
                                       const map_array_t<S, I> *arg1,
                                       const map_array_t<S, I> *arg2)
{
    // XXX maybe generalize to $alpha f(x, y) + beta y$
    if(!self)
        throw std::invalid_argument("apply_binop: self is NULL");
    if(!arg1)
        throw std::invalid_argument("apply_binop: arg1 is NULL");
    if(!arg2)
        throw std::invalid_argument("apply_binop: arg2 is NULL");

    // check that the dimensions are compatible
    for(size_t j=0; j<self->ndim(); ++j){
        if(arg1->shape()[j] != arg2->shape()[j]){
            throw std::invalid_argument("Binop: incompatible dimensions.");
        }
    }

    // result shape is known, set it right away
    self->set_shape(arg1->shape());

    // run over the nonzero elements of *arg1. This is O(n1 * log(n2) * log(n1))
    std::pair<typename map_array_t<T, I>::iterator, bool> p;
    typename map_array_t<T, I>::index_type idx;
    typename map_array_t<S, I>::const_iterator it, it2;
    for(it = arg1->begin(); it != arg1->end(); ++it){
        idx = it->first;
        it2 = arg2->find(idx);
        S y = (it2 != arg2->end()) ? it2->second
                                   : arg2->fill_value();

        T value = (*binop)(it->second, y);

        p = self->insert(std::make_pair(idx, value));
        if(!(p.second)){ p.first->second = value; }   // overwrite what was there at index idx
    }

    if(arg2 != arg1){
        // run over the nonzero elements of *arg2; those which are present in both
        // *arg1 and *arg2 have been taken care of already.
        // This loop's complexity is O(n2 * log(n1) * log(n2))
        std::pair<typename map_array_t<T, I>::iterator, bool> p;
        typename map_array_t<T, I>::index_type idx;
        typename map_array_t<S, I>::const_iterator it1, it2;
        for(it2 = arg2->begin(); it2 != arg2->end(); ++it2){
                idx = it2->first;
                it1 = arg1->find(idx);
                if (it1 == arg1->end()){
                    // it1->second is present in arg2 but not in arg1
                    T value = (*binop)(arg1->fill_value(), it2->second);

                    p = self->insert(std::make_pair(idx, value));
                    if(!(p.second)){ p.first->second = value; }

                }
        }
    }

    // update fill_value
    T fill_value = (*binop)(arg1->fill_value(), arg2->fill_value());
    self->set_fill_value(fill_value);
}


template<typename T, typename I>
inline void
operations<T, I>::apply_mmul(map_array_t<T, I> *C,
                                      const map_array_t<T, I> *A,
                                      const map_array_t<T, I> *B)
{
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
    I arr[2];
    arr[0] = A->shape()[0];
    arr[1] = B->shape()[1];
    C->set_shape(arr);

    // XXX non-zero fill_values are not implemented. The result is likely
    // dense anyway, so having to convert to dense explicitly is not *that* bad.
    if((A->fill_value() != 0) || (B->fill_value() != 0))
        throw std::runtime_error("Non-zero fill_values not handled yet.");

    // C_{ik} = alpha A_{ij} B_{jk} + beta C_{ik}
    std::pair<typename map_array_t<T, I>::iterator, bool> p;
    typename map_array_t<T, I>::iterator itC;
    typename map_array_t<T, I>::const_iterator itA, itB;
    typename map_array_t<T, I>::index_type idxA, idxB;

    T Aij, Bjk, Cik, value;
    for (itA = A->begin(); itA != A->end(); ++itA){
        Aij = itA->second;
        idxA = itA->first;
        arr[0] = idxA[0];
        for(itB = B->begin(); itB != B->end(); ++itB){
            idxB = itB->first;
            if( idxA[1] != idxB[0])
                continue;                       // XXX row/column_iterators?

            Bjk = itB->second;
            arr[1] = idxB[1];
            typename map_array_t<T, I>::index_type idx(arr);
            itC = C->find(idx);
            Cik = (itC != C->end()) ? itC->second
                                    : C->fill_value();
            value = Aij*Bjk + Cik;

            p = C->insert(std::make_pair(idx, value));
            if(!(p.second)){ p.first->second = value; }
        }
    }
}



/*    
 * shape() might be too large if some elements were deleted.
 * Recompute the minimum size shape. Complexity is O(N).
 */
template<typename T, typename I>
inline typename map_array_t<T, I>::index_type
get_min_shape(const map_array_t<T, I>& arg)
{
    typename map_array_t<T, I>::index_type sh;
    for (size_t j = 0; j < arg.ndim(); ++j){ sh[j] = 0; }

    typename map_array_t<T, I>::const_iterator it = arg.begin();
    for (; it != arg.end(); ++it){
        for (size_t j=0; j < arg.ndim(); ++j){ 
            if (it->first[j] >= sh[j]){
                sh[j] = it->first[j] + 1;
            }
        }
    }
    return sh;
}


// TODO:  
//        4. slicing
//        5. special-case zero fill_value (memset, also matmul?)
//        6. flat indexing for d != 2

} // namespace sparray

template<typename T, typename I>
std::ostream&
operator<<(std::ostream& out, const sparray::map_array_t<T, I>& ma)
{
    out << "\n*** shape = " << ma.shape() << "  num_elem = "<< ma.count_nonzero() <<" {\n";
    typename sparray::map_array_t<T, I>::const_iterator it = ma.begin();
    for (; it != ma.end(); ++it){
        std::cout << "    " << it->first << " -> " << it->second <<"\n";
    }
    return out << "}  w/ fill_value = " << ma.fill_value() << "\n";
}

#endif
