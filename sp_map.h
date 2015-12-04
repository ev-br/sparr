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
    void copy_from_other(const map_array_t *p_other);

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

    // indexing helpers
    single_index_type _flat_index(const index_type& index) const;

    // FIXME: return error codes / raise exceptions. Below and elsewhere. 
    // convert to a dense representation (C order). Caller's responsible for
    // allocating memory.
    void todense(void* dest, const single_index_type len) const;

    // elementwise operations
    void inplace_unary_op(T (*fptr)(T x, T a, T b), T a, T b);  // x <- f(x, a, b)
    void inplace_binary_op(T (*fptr)(T x, T y, T a, T b),
                           const map_array_t<T, I, num_dim>* other,
                           T a,
                           T b); // x <- f(x, y, a, b), e.g. x <- a*x + b*y

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
    copy_from_other(&other);
}

// NB This could have been const map_array_t<...>&, but Cython wrappers
// operate on pointers anyway.
template<typename T, typename I, size_t num_dim>
inline void 
map_array_t<T, I, num_dim>::copy_from_other(const map_array_t<T, I, num_dim> *p_other)
{
    if(!p_other)
        throw std::logic_error("copy from NULL");
    if(p_other == this)
        throw std::logic_error("copy from self.");

    data_.insert(p_other->begin_nonzero(), p_other->end_nonzero());
    for(size_t j=0; j < num_dim; ++j){
        shape_[j] = p_other->shape()[j];
    }
    fill_value_ = p_other->fill_value();
}


template<typename T, typename I, size_t num_dim>
inline T
map_array_t<T, I, num_dim>::get_one(const map_array_t::index_type& idx) const
{
    iter_nonzero_type it = data_.find(idx);

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
inline single_index_type
map_array_t<T, I, num_dim>::_flat_index(const typename map_array_t<T, I, num_dim>::index_type& index) const
{
    assert(num_dim == 2);
    single_index_type stride = shape_[1];
    return index[0]*stride + index[1];
}


template<typename T, typename I, size_t num_dim>
inline void
map_array_t<T, I, num_dim>::todense(void* dest, const single_index_type num_elem) const
{
    if (num_elem < 0){ throw std::runtime_error("num_elem < 0"); }
    if (num_elem == 0){ return; }

    // fill the background
    T *_dest = static_cast<T*>(dest);
    std::fill(_dest, _dest + num_elem, fill_value_);

    // fill nonzero elements
    map_array_t<T, I, num_dim>::iter_nonzero_type it = begin_nonzero();
    for(; it != end_nonzero(); ++it){
        single_index_type idx = _flat_index(it->first);
        assert(idx < num_elem);
        _dest[idx] = it->second;
    }
}


// TODO:  
//        2. boolean ops a la numpy: return map_array_t of booleans (allocation!)
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
