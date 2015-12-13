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
    typedef typename map_type::value_type value_type;    // pair<key, value>
    typedef typename map_type::key_type index_type;
    typedef typename map_type::const_iterator const_iterator;
    typedef typename map_type::iterator iterator;

    map_array_t(const T& fill_value = 0);
    map_array_t(const map_array_t& other);
    template<typename S> void copy_from(const map_array_t<S, I, num_dim> *src);

    size_t ndim() const { return num_dim; }
    index_type shape() const { return shape_; }
    void set_shape(const index_type& idx);

    T fill_value() const { return fill_value_; }
    void set_fill_value(const T& value) { fill_value_ = value; }

    // access nonzero elements
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
 
    // convert to a dense representation (C order). Caller's responsible for
    // allocating memory.
    void todense(void* dest, const I len) const;

    //// elementwise operations

    // x <- alpha f(x) + beta x
    void inplace_unary(T (*op)(T x), const T alpha=1, const T beta=0);

    // x <- alpha f(x, y) + beta x, \foreach x in self, \foreach y in other
    void inplace_binop(T (*binop)(T x, T y),
                           const map_array_t<T, I, num_dim>* other,
                           const T alpha=1,
                           const T beta=0);

    // z <- alpha f(x, y) + beta x, \foreach x in arg1, \foreach y in arg2
    template<typename S> void apply_binop(T (*binop)(S x, S y),
                                          const map_array_t<S, I, num_dim>* arg1,
                                          const map_array_t<S, I, num_dim>* arg2);

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
        throw std::invalid_argument("copy from NULL");

    typename map_array_t<S, I, num_dim>::const_iterator it = src->begin();
    typename map_array_t<T, I, num_dim>::iterator hint = this->begin();

    for(; it != src->end(); ++it){
        T value = static_cast<T>(it->second);
        hint = this->insert(hint, std::make_pair(it->first, value));
        hint->second = value;
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
    const_iterator it = this->find(idx);

    if (it == this->end()){
        return fill_value_;
    }
    return it->second;
}


template<typename T, typename I, size_t num_dim>
inline void
map_array_t<T, I, num_dim>::set_one(const map_array_t::index_type& idx, const T& value)
{
    // XXX Can use a hint if inserting consequtive values.
    std::pair<iterator, bool> p = this->insert(std::make_pair(idx, value));
    p.first->second = value;

    // update the shape if needed
    for(size_t j=0; j < num_dim; ++j){
        if(idx[j] >= shape_[j]){
            shape_[j] = idx[j] + 1;
        }
    }
}


template<typename T, typename I, size_t num_dim>
inline void
map_array_t<T, I, num_dim>::inplace_unary(T (*op)(T x), T alpha, T beta)
{
    iterator it = this->begin();
    for (; it != this->end(); ++it){
        it->second = alpha * (*op)(it->second) + beta * it->second;
    }
    fill_value_ = alpha* (*op)(fill_value_) + beta * fill_value_;
}


template<typename T, typename I, size_t num_dim>
inline void
map_array_t<T, I, num_dim>::inplace_binop(T (*binop)(T x, T y),
                                              const map_array_t<T, I, num_dim> *other,
                                              T alpha,
                                              T beta)
{
    if(!other)
        throw std::invalid_argument("binop from NULL");

    // check that the dimensions are compatible
    for(size_t j=0; j<num_dim; ++j){
        if(shape_[j] != other->shape()[j]){
            // TODO: probably want to output the dimensions here
            throw std::invalid_argument("Binop: incompatible dimensions.");
        }
    }

    // run over the nonzero elements of *this. This is O(n1 * log(n2))
    iterator it = this->begin();
    for(; it != this->end(); ++it){
        index_type idx = it->first;
        T y = other->get_one(idx);           // NB: may equal other.fill_value
        it->second = alpha * (*binop)(it->second, y) + beta * it->second;
    }

    if(other != this){
        // run over the nonzero elements of *other; those which are present in both
        // *this and *other have been taken care of already. Insert new ones
        // into *this. This loop's complexity is O(n2 * log(n1))
        const_iterator it_other = other->begin();
        for(; it_other != other->end(); ++it_other){
            index_type idx = it_other->first;
            T value = alpha * (*binop)(fill_value_, it_other->second) + beta * fill_value_;
            this->insert(std::make_pair(idx, value));
        }
    }

    // update fill_value
    fill_value_ = alpha * (*binop)(fill_value_, other->fill_value()) + beta * fill_value_;
}


template<typename T, typename I, size_t num_dim>
template<typename S>
inline void 
map_array_t<T, I, num_dim>::apply_binop(T (*binop)(S x, S y),
                                        const map_array_t<S, I, num_dim>* arg1,
                                        const map_array_t<S, I, num_dim>* arg2)
{
    // XXX maybe generalize to $alpha f(x, y) + beta y$
    if(!arg1)
        throw std::invalid_argument("apply_binop: arg1 is NULL");
    if(!arg2)
        throw std::invalid_argument("apply_binop: arg2 is NULL");

    // check that the dimensions are compatible
    for(size_t j=0; j<num_dim; ++j){
        if(arg1->shape()[j] != arg2->shape()[j]){
            throw std::invalid_argument("Binop: incompatible dimensions.");
        }
    }

    // run over the nonzero elements of *arg1. This is O(n1 * log(n2) * log(n1))
    typename map_array_t<S, I, num_dim>::const_iterator it;
    for(it = arg1->begin(); it != arg1->end(); ++it){
        index_type idx = it->first;
        S y = arg2->get_one(idx);           // NB: may equal other.fill_value
        T value = (*binop)(it->second, y);
        this->set_one(idx, value);
    }

    if(arg2 != arg1){
        // run over the nonzero elements of *arg2; those which are present in both
        // *arg1 and *arg2 have been taken care of already.
        // This loop's complexity is O(n2 * log(n1) * log(n2))
        typename map_array_t<S, I, num_dim>::const_iterator it1, it2;
        for(it2 = arg2->begin(); it2 != arg2->end(); ++it2){
                index_type idx = it2->first;
                it1 = arg1->find(idx);
                if (it1 == arg1->end()){
                    // it1->second is present in arg2 but not in arg1
                    T value = (*binop)(arg1->fill_value(), it2->second);
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
        throw std::invalid_argument("apply_binop: A is NULL");
    if(!B)
        throw std::invalid_argument("apply_binop: B is NULL");

    assert(num_dim == 2);

    if(A->shape()[1] != B->shape()[0])
        throw std::invalid_argument("GEMM: incompatible dimensions.");


    // XXX non-zero fill_values are not implemented. The result is likely
    // dense anyway, so having to convert to dense explicitly is not *that* bad.
    if((A->fill_value() != 0) || (B->fill_value() != 0))
        throw std::runtime_error("Non-zero fill_values not handled yet.");

    // C_{ik} = alpha A_{ij} B_{jk} + beta C_{ik}
    typename map_array_t<T, I, num_dim>::const_iterator itA, itB;
    typename map_array_t<T, I, num_dim>::index_type idxA, idxB;

    T Aij, Bjk, Cik, value;
    I arr[num_dim];
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
            typename map_array_t<T, I, num_dim>::index_type idx(arr);
            Cik = this->get_one(idx);
            value = alpha*Aij*Bjk + beta*Cik;
            this->set_one(idx, value);
        }
    }
}


template<typename T, typename I, size_t num_dim>
inline void 
map_array_t<T, I, num_dim>::set_shape(const index_type& shp)
{
    index_type min_shape = get_min_shape(*this);

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
    const_iterator it = begin();
    for(; it != end(); ++it){
        I idx = _flat_index(it->first);
        assert(idx < num_elem);
        _dest[idx] = it->second;
    }
}

/*    
 * shape() might be too large if some elements were deleted.
 * Recompute the minimum size shape. Complexity is O(N).
 */
template<typename T, typename I, size_t num_dim>
inline typename map_array_t<T, I, num_dim>::index_type
get_min_shape(const map_array_t<T, I, num_dim>& arg)
{
    typename map_array_t<T, I, num_dim>::index_type sh;
    for (size_t j = 0; j < num_dim; ++j){ sh[j] = 0; }

    typename map_array_t<T, I, num_dim>::const_iterator it = arg.begin();
    for (; it != arg.end(); ++it){
        for (size_t j=0; j < num_dim; ++j){ 
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

template<typename T, typename I, size_t num_dim>
std::ostream&
operator<<(std::ostream& out, const sparray::map_array_t<T, I, num_dim>& ma)
{
    out << "\n*** shape = " << ma.shape() << "  num_elem = "<< ma.count_nonzero() <<" {\n";
    typename sparray::map_array_t<T, I, num_dim>::const_iterator it = ma.begin();
    for (; it != ma.end(); ++it){
        std::cout << "    " << it->first << " -> " << it->second <<"\n";
    }
    return out << "}  w/ fill_value = " << ma.fill_value() << "\n";
}

#endif
