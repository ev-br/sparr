#ifndef SP_FIXED_CAPACITY_H
#define SP_FIXED_CAPACITY_H
#include<assert.h>
#include<iostream>
#include<string>

namespace sparray {

typedef long int single_index_type;  // FIXME: npy_intp, likely Py_ssize_t


template<typename I=single_index_type, size_t num_dim=2>
struct fixed_capacity
{
    I elem_[num_dim];

    I& operator[](size_t j){
        assert(j < num_dim);
        return elem_[j];
    }
    const I& operator[](size_t j) const{
        assert(j < num_dim);
        return elem_[j];
    };

    fixed_capacity(const I* c_arr = NULL);  // NB: no range checking

};
// XXX: zero-initialize?


template<typename I, size_t num_dim>
fixed_capacity<I, num_dim>::fixed_capacity(const I* c_arr)
{
    if (c_arr){
        std::copy(c_arr, c_arr + num_dim, elem_);
    }
}


/* Comparator for std::map<fixed_capacity, ...>*/
template<typename I=single_index_type, size_t num_dim=2>
struct fixed_capacity_cmp 
{
    bool operator()(const fixed_capacity<I, num_dim>& lhs,
                    const fixed_capacity<I, num_dim>& rhs) const
    {
        for (size_t j=0; j < num_dim; ++j){
            I l = lhs.elem_[j], r = rhs.elem_[j];
            if(l > r){ return false; }
            else if (l < r){ return true; }
        }
        return false;
    }
};

#define FC_ELEM(fc, j) (fc).elem_[(j)]
#define FC_ELEMS(fc) (fc).elem_


std::string bool_outp(const bool& x){ return x ? "true" : "false"; }


} // namespace sparray


template<typename I, size_t num_dim>
std::ostream&
operator<<(std::ostream& out, const sparray::fixed_capacity<I, num_dim>& v)
{
    out << "["; 
    for (size_t j=0; j < num_dim; ++j){
        out << v.elem_[j];
        if (j < num_dim - 1){ out << ", ";}
    }
    return out << "]";
}

#endif
