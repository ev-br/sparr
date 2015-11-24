#ifndef SP_MAP_H
#define SP_MAP_H
#include<map>
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
    void copy_from_other(const map_array_t& other);

    size_t ndim() const { return num_dim; }
    index_type shape() const { return shape_; }

    // shape() might be too large if some elements were deleted
    // this recomputes it, which is O(N).
    index_type get_min_shape() const;

    T fill_value() const { return fill_value_; }
    void set_fill_value(const T& value) { fill_value_ = value; }

    // access nonzero elements
    size_t count_nonzero() const { return data_.size(); }
    iter_nonzero_type begin_nonzero() const { return data_.begin(); }
    iter_nonzero_type end_nonzero() const { return data_.end(); }

    // retrieve / set a single element
    T get_one(const index_type& idx) const;
    void set_one(const index_type& idx, const T& value);

    // FIXME: return error codes / raise exceptions. Below and elsewhere. 

    // convert to a dense representation (C order). Caller's responsible for allocations.
    void todense(T* dest, const int len) const;

    // elementwise operations
    void inplace_unary_op(T (*fptr)(T x, T a, T b), T a, T b);  // x <- f(x, a, b)

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
    copy_from_other(other);
}


template<typename T, typename I, size_t num_dim>
inline void 
map_array_t<T, I, num_dim>::copy_from_other(const map_array_t<T, I, num_dim>& other)
{
    data_.insert(other.begin_nonzero(), other.end_nonzero());
    for(size_t j=0; j < num_dim; ++j){
        shape_[j] = other.shape()[j];
    }
    fill_value_ = other.fill_value();
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
inline typename map_array_t<T, I, num_dim>::index_type
map_array_t<T, I, num_dim>::get_min_shape() const
{
    index_type sh;
    for (size_t j = 0; j < num_dim; ++j){ sh[j] = 0; }

    iter_nonzero_type it = this->begin_nonzero();
    for (; it != this->end_nonzero(); ++it){
        for (size_t j=0; j < num_dim; ++j){ 
            if (it->second[j] >= sh[j]){
                sh[j] = it->second[j] + 1;
            }
        }
    }
    return sh;
}

// TODO: 1. binary ops
//        2. boolean ops a la numpy: return map_array_t of booleans (allocation!)
//        3. todense
//        4. slicing

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
    return out << "}";
}

#endif
