#ifndef SP_VIEW_H
#define SP_VIEW_H
#include <stdexcept>

#include <boost/iterator.hpp>
#include <boost/iterator/iterator_adaptor.hpp>

#include "sp_map.h"

/*
 * Notation: given a = [1, 2, 3, 4], the indices to `a` are prefixed with m,
 * like so: mIdx, and indices to a[::2] are prefixed with v, like so: vIdx. 
 */

namespace sparray {

/* 
 * Store the result of:
 *      PySlice_IndicesEx,
 *      reduce start and stop to += length, if necessary
 */
struct slice_t
{
    single_index_type start, stop, step;
    single_index_type slicelength;
};


/*
 * Check is an mIdx is contained in the slice, and return the corresponding vIdx.
 * If mIdx is invalid (i.e., not in slice), vIdx is garbage.
 */
inline std::pair<bool, single_index_type>
vIdx_from_mIdx(const slice_t& slice, single_index_type mIdx) {
    if (slice.slicelength == 0) {
        return std::make_pair(false, -1);
    }
    single_index_type num, rem;
    num = (mIdx - slice.start) / slice.step;
    rem = (mIdx - slice.start) % slice.step;
    bool is_valid = (rem == 0) && (num >= 0) && (num < slice.slicelength);
    return std::make_pair(is_valid, num);
}


/*
 * Helper for the strided_iterator: check if mIdx is in slice and if it is,
 * return the corresponding vIdx.
 * Here mIdx and vIdx are map_t::index_types.
 */
template<typename Key>
struct converter_t
{
    std::vector<slice_t> m_slices;

    converter_t(const std::vector<slice_t>& slices)
        : m_slices(slices) {}

    std::pair<bool, Key> v_from_m(const Key& x) const;

///    bool is_divisible(const map_t::value_type& x) const { return (x.first % m_n == 0); }
///    map_t::value_type convert_index(const map_t::value_type& x) const { 
///        return std::make_pair(x.first / m_n, x.second); 
///    }
};

template<typename Key>
std::pair<bool, Key>
converter_t<Key>::v_from_m(const Key& mIdx) const
{
    assert(mIdx.ndim() == m_slices.size());

    Key vIdx(m_slices.size());
    std::pair<bool, single_index_type> p;

    for (size_t j=0; j < m_slices.size(); j++) {
        p = vIdx_from_mIdx(m_slices[j], mIdx[j]);
        if (!p.first) {
            // mIdx is not in slice, can return garbage for vIdx
            return std::make_pair(false, vIdx);
        }
        vIdx[j] = p.second;
    }
    return std::make_pair(true, vIdx);
}


/////////////////////////////////////////////////// the iterator
template<typename Iterator, typename Key, typename Value>
class slice_iterator
    : public boost::iterator_adaptor<
        slice_iterator<Iterator, Key, Value>,     // Derived
        Iterator,             // Base: map_t::iterator
        Value,                // value: map_t::value_type
        boost::iterators::use_default,
        Value                 // reference  ??? HACK ???
      >
{
    public:
        slice_iterator()
            : slice_iterator::iterator_adaptor_(),
              m_end(),
              m_conv() {}

        explicit slice_iterator(const Iterator& p,
                                  const Iterator& end_,
                                  const converter_t<Key>& conv)
            : slice_iterator::iterator_adaptor_(p),
              m_end(end_),
              m_conv(conv) {}

    private:
        friend class boost::iterator_core_access;

        Iterator m_end;
        converter_t<Key> m_conv;

        typename slice_iterator<Iterator, Key, Value>::reference dereference() const {
            std::pair<bool, Key> p_vIdx = m_conv.v_from_m(this->base()->first);
            assert(p_vIdx.first);
            return std::make_pair(p_vIdx.second, this->base()->second);
        }

        // copied almost verbatim from filter_iterator.hpp
        void increment() {
            ++(this->base_reference());
            satisfy_predicate();
           // XXX can store the vIdx from m_conv.v_from_m for reuse in dereference
        }

        void satisfy_predicate() {
            while (this->base() != this->m_end &&
                   !this->m_conv.v_from_m(this->base()->first).first) {
                ++(this->base_reference());
            }
        }

        void decrement() {
            // FIXME take eg from filter_iterator.hpp
            throw std::runtime_error("decrement");
        }
};

/////////////////////////////////////////////////// the view
template<typename T, typename I=single_index_type>
struct view_t
{

    private:
        map_array_t<T, I> *m_map;
        std::vector<slice_t> m_slices;
};


} // end namespace sparray

#endif
