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
    bool m_initd;

    fixed_capacity(int ndim=0) {
        m_initd = false;
        if (ndim > 0) {
            m_elem.resize(ndim);
            m_initd = true;
        }
    }

    I& operator[](size_t j){
        assert(j < m_elem.size());
        assert(m_initd);
        return m_elem[j];
    }
    const I& operator[](size_t j) const{
        assert(j < m_elem.size());
        assert(m_initd);
        return m_elem[j];
    };

    //int ndim() const { assert(m_initd); return (int)m_elem.size(); }

};

typedef fixed_capacity<single_index_type> index_type;


/* Comparator for std::map<fixed_capacity, ...>*/
template<typename I=single_index_type>
struct fixed_capacity_cmp 
{
    int m_ndim;

    fixed_capacity_cmp(int num_dim) : m_ndim(num_dim) {}
    bool operator()(const fixed_capacity<I>& lhs,
                    const fixed_capacity<I>& rhs) const
    {
        assert((lhs.m_elem.size() == rhs.m_elem.size()) 
               && ((int)lhs.m_elem.size() == m_ndim));      // XXX

        for (int j=0; j < m_ndim; ++j){
            I l = lhs[j], r = rhs[j];
            if(l > r){ return false; }
            else if (l < r){ return true; }
        }
        return false;
    }
};

std::string bool_outp(const bool& x){ return x ? "true" : "false"; }


// The factory which knows the ndim and creates instances of fixed_capacity
// of this size.
template<typename I=single_index_type>
struct fixed_capacity_factory_t
{
    fixed_capacity_factory_t(const int num_dim) : m_ndim(num_dim) {}

    int ndim() const {return m_ndim;}

    fixed_capacity<I> get_new(){
        return fixed_capacity<I>(m_ndim);
    } 

    private:
        int m_ndim;
};

typedef fixed_capacity_factory_t<single_index_type> index_factory_t;


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
