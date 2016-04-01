#ifndef SP_FIXED_CAPACITY_H
#define SP_FIXED_CAPACITY_H
#include<assert.h>
#include<iostream>
#include <vector>
#include<string>
#include"common_types.h"

namespace sparray {


template<typename I=single_index_type>
struct fixed_capacity
{
    std::vector<I> m_elem;

    fixed_capacity(int ndim=2) {m_elem.resize(ndim); }

    I& operator[](size_t j){
        assert(j < m_elem.size());
        return m_elem[j];
    }
    const I& operator[](size_t j) const{
        assert(j < m_elem.size());
        return m_elem[j];
    };

    int ndim() const { return (int)m_elem.size(); }

};

typedef fixed_capacity<single_index_type> index_type;


/* Comparator for std::map<fixed_capacity, ...>*/
template<typename I=single_index_type>
struct fixed_capacity_cmp 
{
    bool operator()(const fixed_capacity<I>& lhs,
                    const fixed_capacity<I>& rhs) const
    {
        assert(lhs.ndim() == rhs.ndim());

        for (int j=0; j < lhs.ndim(); ++j){
            I l = lhs[j], r = rhs[j];
            if(l > r){ return false; }
            else if (l < r){ return true; }
        }
        return false;
    }
};

std::string bool_outp(const bool& x){ return x ? "true" : "false"; }


} // namespace sparray


#define FC_ELEM(fc, j) (fc).m_elem[(j)]
#define FC_ELEMS(fc) &((fc).m_elem)[0]


template<typename I>
std::ostream&
operator<<(std::ostream& out, const sparray::fixed_capacity<I>& v)
{
    out << "["; 
    for (size_t j=0; j < v.ndim(); ++j){
        out << v[j];
        if (j < v.ndim() - 1){ out << ", ";}
    }
    return out << "]";
}

#endif
