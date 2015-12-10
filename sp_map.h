#ifndef SP_MAP_H
#define SP_MAP_H
#include<map>
#include<stdexcept>
#include"fixed_cap.h"


namespace sparray {


template<typename T, typename I=single_index_type, size_t num_dim=2>
struct map_array_t
{
    typedef std::map<fixed_capacity<I,num_dim>,
                      T,
                      fixed_capacity_cmp<I, num_dim> > map_type;
    typedef typename map_type::mapped_type value_type;
    typedef typename map_type::key_type index_type;
    typedef typename map_type::const_iterator iter_nonzero_type;

    map_array_t(const T& fill_value = 0);
    map_array_t(const map_array_t& other);
    template<typename S> void copy_from(const map_array_t<S, I, num_dim> *src);

    size_t ndim() const { return num_dim; }
    index_type shape() const { return shape_; }

    // shape() might be too large if some elements were deleted
    // this recomputes it, which is O(N).
    index_type get_min_shape() const;

    // set the shape
    void set_shape(const index_type& idx);
    T fill_value() const { return fill_value_; }
    void set_fill_value(const T& value) { fill_value_ = value; }

    // access nonzero elements
    size_t count_nonzero() const { return data_.size(); }
    iter_nonzero_type begin_nonzero() const { return data_.begin(); }
    iter_nonzero_type end_nonzero() const { return data_.end(); }

    // retrieve / set a single element
    T get_one(const index_type& idx) const;
    void set_one(const index_type& idx, const T& value);

    iter_nonzero_type find(const index_type& idx) const { return data_.find(idx);}

    // indexing helpers
    I _flat_index(const index_type& index) const;

    // FIXME: return error codes / raise exceptions. Below and elsewhere. 
    // convert to a dense representation (C order). Caller's responsible for
    // allocating memory.
    void todense(void* dest, const I len) const;

    // elementwise operations
    void inplace_unary_op(T (*fptr)(T x, T a, T b), T a, T b);  // x <- f(x, a, b)
    void inplace_binary_op(T (*fptr)(T x, T y, T a, T b),
                           const map_array_t<T, I, num_dim>* other,
                           T a,
                           T b); // x <- f(x, y, a, b), e.g. x <- a*x + b*y

    template<typename S> void apply_binop(T (*binop)(S x, S y),
                                          const map_array_t<S, I, num_dim>* first,
                                          const map_array_t<S, I, num_dim>* second);

    // GEMM: C <- alpha A @ B + beta C 
    void inplace_gemm(const T alpha,
                      const map_array_t<T, I, num_dim>* A,
                      const map_array_t<T, I, num_dim>* B,
                      const T beta);

    private:
        map_type data_;
        index_type shape_;
        T fill_value_;
};


template<typename T, typename I, size_t num_dim>
inline map_array_t<T, I, num_dim>::map_array_t(const T& fill_value)
{
    for(size_t j=0; j < num_dim; ++j){
        shape_[j] = 0;
    }
    fill_value_ = fill_value;
}


template<typename T, typename I, size_t num_dim>
inline map_array_t<T, I, num_dim>::map_array_t(const map_array_t<T, I, num_dim>& other)
{
    copy_from(&other);
}

// NB This could have been const map_array_t<...>&, but Cython wrappers
// operate on pointers anyway.
template<typename T, typename I, size_t num_dim>
template<typename S>
inline void
map_array_t<T, I, num_dim>::copy_from(const map_array_t<S, I, num_dim> *src)
{
    if(!src)
        throw std::logic_error("copy from NULL");

    typename map_array_t<S, I, num_dim>::iter_nonzero_type it = src->begin_nonzero();

    for(; it != src->end_nonzero(); ++it){
        data_[it->first] = static_cast<T>(it->second);
    }

    for(size_t j=0; j < num_dim; ++j){
        shape_[j] = src->shape()[j];
    }

    fill_value_ = static_cast<T>(src->fill_value());
}


template<typename T, typename I, size_t num_dim>
inline T
map_array_t<T, I, num_dim>::get_one(const map_array_t::index_type& idx) const
{
    iter_nonzero_type it = this->find(idx);

    if (it == data_.end()){
        return fill_value_;
    }
    return it->second;
}


template<typename T, typename I, size_t num_dim>
inline void
map_array_t<T, I, num_dim>::set_one(const map_array_t::index_type& idx, const T& value)
{
    
    data_[idx] = value;

    // update the shape if needed
    for(size_t j=0; j < num_dim; ++j){
        if(idx[j] >= shape_[j]){
            shape_[j] = idx[j] + 1;
        }
    }
}


template<typename T, typename I, size_t num_dim>
inline void
map_array_t<T, I, num_dim>::inplace_unary_op(T (*fptr)(T x, T a, T b), T a, T b)
{
    for (typename map_type::iterator it = data_.begin();
         it != data_.end();
         ++it){
        it->second = (*fptr)(it->second, a, b);
    }
    fill_value_ = (*fptr)(fill_value_, a, b);
}


template<typename T, typename I, size_t num_dim>
inline void
map_array_t<T, I, num_dim>::inplace_binary_op(T (*fptr)(T x, T y, T a, T b),
                                              const map_array_t<T, I, num_dim> *p_other,
                                              T a,
                                              T b)
{
    if(!p_other)
        throw std::logic_error("binop from NULL");

    // check that the dimensions are compatible
    for(size_t j=0; j<num_dim; ++j){
        if(shape_[j] != p_other->shape()[j]){
            // TODO: probably want to output the dimensions here
            throw std::invalid_argument("Binop: incompatible dimensions.");
        }
    }

    // run over the nonzero elements of *this. This is O(n1 * log(n2))
    typename map_type::iterator it = data_.begin();
    for(; it != data_.end(); ++it){
        index_type idx = it->first;
        T y = p_other->get_one(idx);           // NB: may equal other.fill_value
        it->second = (*fptr)(it->second, y, a, b);
    }

    if(p_other != this){
        // run over the nonzero elements of *other; those which are present in both
        // *this and *other have been taken care of already. Insert new ones
        // into *this. This loop's complexity is O(n2 * log(n1))
        iter_nonzero_type it_other = p_other->begin_nonzero();
        for(; it_other != p_other->end_nonzero(); ++it_other){
            index_type idx = it_other->first;
            it = data_.find(idx);
            if (it == data_.end()){
                it->second = (*fptr)(fill_value_, it_other->second, a, b);
            }
        }
    }

    // update fill_value
    fill_value_ = (*fptr)(fill_value_, p_other->fill_value(), a, b);
}


template<typename T, typename I, size_t num_dim>
template<typename S>
inline void 
map_array_t<T, I, num_dim>::apply_binop(T (*binop)(S x, S y),
                                        const map_array_t<S, I, num_dim>* arg1,
                                        const map_array_t<S, I, num_dim>* arg2)
{
    if(!arg1)
        throw std::logic_error("apply_binop: arg1 is NULL");
    if(!arg2)
        throw std::logic_error("apply_binop: arg2 is NULL");

    // check that the dimensions are compatible
    for(size_t j=0; j<num_dim; ++j){
        if(arg1->shape()[j] != arg2->shape()[j]){
            throw std::invalid_argument("Binop: incompatible dimensions.");
        }
    }

    // run over the nonzero elements of *arg1. This is O(n1 * log(n2) * log(n1))
    typename map_array_t<S, I, num_dim>::iter_nonzero_type it;
    for(it = arg1->begin_nonzero();
        it != arg1->end_nonzero();
        ++it){
            index_type idx = it->first;
            S y = arg2->get_one(idx);           // NB: may equal other.fill_value
            T value = (*binop)(it->second, y);
            this->set_one(idx, value);
    }

    if(arg2 != arg1){
        // run over the nonzero elements of *arg2; those which are present in both
        // *arg1 and *arg2 have been taken care of already.
        // This loop's complexity is O(n2 * log(n1) * log(n2))
        typename map_array_t<S, I, num_dim>::iter_nonzero_type it1, it2;
        for(it2 = arg2->begin_nonzero();
            it2 != arg2->end_nonzero();
            ++it2){
                index_type idx = it2->first;
                it1 = arg1->find(idx);
                if (it1 == arg1->end_nonzero()){
                    // it1->second is present in arg2 but not in arg1
                    T value = (*binop)(it1->second, it2->second);
                    this->set_one(idx, value);
                }
        }
    }

    // update fill_value
    fill_value_ = (*binop)(arg1->fill_value(), arg2->fill_value());

}


template<typename T, typename I, size_t num_dim>
inline void
map_array_t<T, I, num_dim>::inplace_gemm(const T alpha,
                                         const map_array_t<T, I, num_dim>* A,
                                         const map_array_t<T, I, num_dim>* B,
                                         const T beta)
{
    if(!A)
        throw std::logic_error("apply_binop: A is NULL");
    if(!B)
        throw std::logic_error("apply_binop: B is NULL");

    assert(num_dim == 2);

    if(A->shape()[1] != B->shape()[0])
        throw std::invalid_argument("GEMM: incompatible dimensions.");


    // XXX non-zero fill_values are not implemented. The result is likely
    // dense anyway, so having to convert to dense explicitly is not *that* bad.
    if((A->fill_value() != 0) || (B->fill_value() != 0))
        throw std::runtime_error("Non-zero fill_values not handled yet.");

    // C_{ik} = alpha A_{ij} B_{jk} + beta C_{ik}
    typename map_array_t<T, I, num_dim>::iter_nonzero_type itA, itB;
    typename map_array_t<T, I, num_dim>::index_type idxA, idxB;

    T Aij, Bjk, Cik, value;
    I arr[num_dim];
    for (itA = A->begin_nonzero(); itA != A->end_nonzero(); ++itA){
        Aij = itA->second - A->fill_value();
        idxA = itA->first;
        arr[0] = idxA[0];
        for(itB = B->begin_nonzero(); itB != B->end_nonzero(); ++itB){
            idxB = itB->first;
            if( idxA[1] != idxB[0])
                continue;                       // XXX row/column_iterators?

            Bjk = itB->second - B->fill_value();
            arr[1] = idxB[1];
            typename map_array_t<T, I, num_dim>::index_type idx(arr);
            Cik = this->get_one(idx);
            value = alpha*Aij*Bjk + beta*Cik;
            this->set_one(idx, value);
        }
    }
}


template<typename T, typename I, size_t num_dim>
inline typename map_array_t<T, I, num_dim>::index_type
map_array_t<T, I, num_dim>::get_min_shape() const
{
    index_type sh;
    for (size_t j = 0; j < num_dim; ++j){ sh[j] = 0; }

    iter_nonzero_type it = this->begin_nonzero();
    for (; it != this->end_nonzero(); ++it){
        for (size_t j=0; j < num_dim; ++j){ 
            if (it->first[j] >= sh[j]){
                sh[j] = it->first[j] + 1;
            }
        }
    }
    return sh;
}


template<typename T, typename I, size_t num_dim>
inline void 
map_array_t<T, I, num_dim>::set_shape(const index_type& shp)
{
    index_type min_shape = get_min_shape();

    for(size_t j = 0; j < num_dim; ++j){
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
template<typename T, typename I, size_t num_dim>
inline I
map_array_t<T, I, num_dim>::_flat_index(const typename map_array_t<T, I, num_dim>::index_type& index) const
{
    assert(num_dim == 2);
    I stride = shape_[1];
    return index[0]*stride + index[1];
}


template<typename T, typename I, size_t num_dim>
inline void
map_array_t<T, I, num_dim>::todense(void* dest, const I num_elem) const
{
    if (num_elem < 0){ throw std::runtime_error("num_elem < 0"); }
    if (num_elem == 0){ return; }

    // fill the background
    T *_dest = static_cast<T*>(dest);
    std::fill(_dest, _dest + num_elem, fill_value_);

    // fill nonzero elements
    map_array_t<T, I, num_dim>::iter_nonzero_type it = begin_nonzero();
    for(; it != end_nonzero(); ++it){
        I idx = _flat_index(it->first);
        assert(idx < num_elem);
        _dest[idx] = it->second;
    }
}


// TODO:  
//        4. slicing
//        5. special-case zero fill_value (memset, also matmul?)
//        6. flat indexing for d != 2
//        8. matmul & inplace axpy

} // namespace sparray

template<typename T, typename I, size_t num_dim>
std::ostream&
operator<<(std::ostream& out, const sparray::map_array_t<T, I, num_dim>& ma)
{
    out << "\n*** shape = " << ma.shape() << "  num_elem = "<< ma.count_nonzero() <<" {\n";
    typename sparray::map_array_t<T, I, num_dim>::iter_nonzero_type it = ma.begin_nonzero();
    for (; it != ma.end_nonzero(); ++it){
        std::cout << "    " << it->first << " -> " << it->second <<"\n";
    }
    return out << "}  w/ fill_value = " << ma.fill_value() << "\n";
}

#endif
