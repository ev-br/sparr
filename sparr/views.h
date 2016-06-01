#ifndef SP_VIEW_H
#define SP_VIEW_H
#include <stdexcept>

#include <boost/iterator.hpp>
#include <boost/iterator/iterator_adaptor.hpp>

#include "common_types.h"
#include "sp_map.h"

/*
 * Notation: given a = [1, 2, 3, 4], the indices to `a` are prefixed with m,
 * like so: mIdx, and indices to a[::2] are prefixed with v, like so: vIdx. 
 */

namespace sparray {


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

    converter_t() {}   // XXX: default ctor is for the identical conversion

    converter_t(const std::vector<slice_t>& slices)
        : m_slices(slices) {}

    std::pair<bool, Key> v_from_m(const Key& mIdx) const;
    Key m_from_v(const Key& vIdx) const;

    bool is_identical() const { return m_slices.empty(); }
};

template<typename Key>
inline std::pair<bool, Key>
converter_t<Key>::v_from_m(const Key& mIdx) const
{
    // XXX: remove this shortcut hack (default ctor means identical transform)
    if (is_identical())
        return std::make_pair(true, mIdx);

    assert(mIdx.ndim() == (int)m_slices.size());

    Key vIdx(m_slices.size());
    std::pair<bool, single_index_type> p;

    for (size_t j=0; j < m_slices.size(); ++j) {
        p = vIdx_from_mIdx(m_slices[j], mIdx[j]);
        if (!p.first) {
            // mIdx is not in slice, can return garbage for vIdx
            return std::make_pair(false, vIdx);
        }
        vIdx[j] = p.second;
    }
    return std::make_pair(true, vIdx);
}

template<typename Key>
inline Key
converter_t<Key>::m_from_v(const Key& vIdx) const
{
    // XXX: remove this shortcut hack (default ctor means identical transform)
    if (is_identical())
        return vIdx;

    assert(vIdx.ndim() == (int)m_slices.size());

    Key mIdx(m_slices.size());
    for (size_t j=0; j < m_slices.size(); ++j) {
        mIdx[j] = m_slices[j].start + vIdx[j]*m_slices[j].step; 
    }
    return mIdx;
}


/////////////////////////////////////////////////// the slice iterator
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

        typedef typename slice_iterator<Iterator, Key, Value>::reference reference;  // ????


    private:
        friend class boost::iterator_core_access;

        Iterator m_end;
        converter_t<Key> m_conv;    // XXX: who manages m_conv: this or the container

        reference dereference() const {
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



////////////////////// the views ////////////////////////////


/*
 * The interface for the views
 */
template<typename T, typename I>
struct abstract_view_t
{
    typedef map_array_t<T, I> Array;
    typedef typename Array::data_type data_type;
    typedef typename Array::value_type value_type;    // pair<key, value>
    typedef typename Array::single_index_type single_index_type;
    typedef typename Array::index_type index_type;
    typedef slice_iterator<typename Array::iterator,
                           typename Array::index_type,
                           typename Array::value_type> iterator;
    typedef slice_iterator<typename Array::const_iterator,
                           typename Array::index_type,
                           typename Array::value_type const> const_iterator;

    // NB: the base pointer is always managed by the caller
    // default ctor is no transform
    abstract_view_t(Array *base)
        : m_base(base),
          m_conv() {}
    abstract_view_t(Array *base,
                    const std::vector<slice_t>& slices)
        : m_base(base),
          m_conv(slices) {}
    abstract_view_t(abstract_view_t *base_view,
                    const std::vector<slice_t>& slices)
        : m_base(base_view->m_base),
          m_conv(slices) {}

    virtual ~abstract_view_t() {}

    virtual int ndim() const = 0;
    virtual index_type shape() const = 0;
    virtual void set_shape(const index_type&) const = 0;

    virtual data_type fill_value() const = 0;
    virtual void set_fill_value(const data_type& value) = 0;

    virtual single_index_type count_nonzero() const = 0;

    virtual const_iterator cbegin() const = 0;
    virtual const_iterator cend() const = 0;

    virtual iterator begin() = 0;
    virtual iterator end() = 0;

    virtual const_iterator find(const index_type& idx) const = 0;
    virtual iterator find(const index_type& idx) = 0;

    // this is *unsafe* because it does NOT update the shape.
    // NB: set_one does.
    virtual std::pair<iterator, bool> _insert(const value_type& val) = 0;

    virtual data_type get_one(const index_type&) const = 0;
    virtual void set_one(const index_type&, const data_type&) = 0;

    // use this instead of `it->second = value`.
    // TODO: `it->second = value` quietly assigns to a temporary :-(
    //   1. double-check for it->second = smth in operations.h
    //   2. does this mean that only const_iterators are needed. 
    virtual void set_one(const iterator&, const data_type&) = 0;

    virtual single_index_type _flat_index(const index_type& vIdx) const = 0;

    std::vector<slice_t> get_slices() const { return m_conv.m_slices; }

    protected:
        Array *m_base;
        converter_t<index_type> m_conv;
};


/*
 * The identical view onto the whole data array 
 */
template<typename T, typename I>
struct map_view_t : public abstract_view_t<T, I>
{
    typedef map_array_t<T, I> Array;
    typedef typename Array::data_type data_type;
    typedef typename Array::value_type value_type;    // pair<key, value>
    typedef typename Array::single_index_type single_index_type;
    typedef typename Array::index_type index_type;
    typedef slice_iterator<typename Array::iterator,
                           typename Array::index_type,
                           typename Array::value_type> iterator;
    typedef slice_iterator<typename Array::const_iterator,
                           typename Array::index_type,
                           typename Array::value_type const> const_iterator;

    map_view_t(Array *base)
        : abstract_view_t<T, I>(base)
    {
        assert(this->m_base);
        assert(this->m_conv.is_identical());
    }
    virtual ~map_view_t() {}

    int ndim() const {
        return this->m_base->ndim();
    }
    index_type shape() const {
        return this->m_base->shape();
    }
    void set_shape(const index_type& shp) const {
        if (shp.ndim() != this->ndim())
            throw std::runtime_error("Cannot change the dimensionality like that.");
        this->m_base->set_shape(shp);
    }

    data_type fill_value() const {
        return this->m_base->fill_value();
    }
    void set_fill_value(const data_type& value) {
        this->m_base->set_fill_value(value);
    }

    single_index_type count_nonzero() const {
        return this->m_base->count_nonzero();
    }

    const_iterator cbegin() const {
        return const_iterator(this->m_base->cbegin(),
                              this->m_base->cend(),
                              this->m_conv); 
    }
    const_iterator cend() const {
        return const_iterator(this->m_base->cend(),
                              this->m_base->cend(),
                              this->m_conv);
    }
    iterator begin() {
        return iterator(this->m_base->begin(),
                        this->m_base->end(),
                        this->m_conv);
    }
    iterator end() {
        return iterator(this->m_base->end(),
                        this->m_base->end(),
                        this->m_conv);
    }

    const_iterator find(const index_type& idx) const {
        return const_iterator(this->m_base->find(idx),
                              this->m_base->cend(),
                              this->m_conv);
    }
    iterator find(const index_type& idx) {
        return iterator(this->m_base->find(idx),
                        this->m_base->end(),
                        this->m_conv);
    }

    std::pair<iterator, bool> _insert(const value_type& val) {
        std::pair<typename Array::iterator, bool> p = this->m_base->_insert(val);
        iterator it(p.first, this->m_base->end(), this->m_conv);
        return std::make_pair(it, p.second);
    }

    data_type get_one(const index_type& idx) const {
        return this->m_base->get_one(idx);
    }
    void set_one(const index_type& idx, const data_type& value) {
        this->m_base->set_one(idx, value);
    }

    void set_one(const iterator& it, const data_type& value) {
        it.base()->second = value;
    }

    virtual single_index_type _flat_index(const index_type& vIdx) const {
        return _flat_index_helper<data_type, single_index_type>(vIdx, shape(), ndim());
    }
};


/*
 * The sliced view: a[::2] et al.
 */
template<typename T, typename I>
struct view_view_t : public abstract_view_t<T, I>
{
    typedef map_array_t<T, I> Array;
    typedef typename Array::data_type data_type;
    typedef typename Array::value_type value_type;    // pair<key, value>
    typedef typename Array::single_index_type single_index_type;
    typedef typename Array::index_type index_type;
    typedef slice_iterator<typename Array::iterator,
                           typename Array::index_type,
                           typename Array::value_type> iterator;
    typedef slice_iterator<typename Array::const_iterator,
                           typename Array::index_type,
                           typename Array::value_type const> const_iterator;

    view_view_t(abstract_view_t<T, I> *base_view,
                const std::vector<slice_t>& slices)
        : abstract_view_t<T, I>(base_view, slices)
    {
        if (!base_view)
            throw std::runtime_error("view: base cannot be NULL");
        assert(base_view->ndim() == (int)slices.size());
    }

    int ndim() const {
        assert(this->m_base);
        return this->m_base->ndim();
    }
    index_type shape() const {
        assert(this->m_base);
        index_type sh(this->m_base->ndim());
        for (size_t j=0; j < this->m_base->ndim(); ++j){
            sh[j] = this->m_conv.m_slices[j].slicelength;
        }
        return sh;
    }
    void set_shape(const index_type&) const {
        throw std::runtime_error("set_shape on a view");
    }

    data_type fill_value() const {
        assert(this->m_base);
        return this->m_base->fill_value();
    }
    void set_fill_value(const data_type& value) {
        assert(this->m_base);
        this->m_base->set_fill_value(value);
    }

    single_index_type count_nonzero() const {
        assert(this->m_base);

        single_index_type count = 0;
        const_iterator it = cbegin();
        for(count = 0; it != cend(); ++it, ++count) {}
        return count;
    }

    const_iterator cbegin() const {
        assert(this->m_base);
        return const_iterator(this->m_base->cbegin(),
                              this->m_base->cend(),
                              this->m_conv); 
    }
    const_iterator cend() const {
        assert(this->m_base);
        return const_iterator(this->m_base->cend(),
                              this->m_base->cend(),
                              this->m_conv);
    }
    iterator begin() {
        assert(this->m_base);
        return iterator(this->m_base->begin(),
                        this->m_base->end(),
                        this->m_conv);
    }
    iterator end() {
        assert(this->m_base);
        return iterator(this->m_base->end(),
                        this->m_base->end(),
                        this->m_conv);
    }

    const_iterator find(const index_type& vIdx) const {
        assert(this->m_base);
        index_type mIdx = this->m_conv.m_from_v(vIdx);
        return const_iterator(this->m_base->find(mIdx), this->m_base->end(), this->m_conv);
    }
    iterator find(const index_type& vIdx) {
        assert(this->m_base);
        index_type mIdx = this->m_conv.m_from_v(vIdx);
        return iterator(this->m_base->find(mIdx), this->m_base->end(), this->m_conv);
    }

    // this is *unsafe*, as this does NOT update the shape.
    // NB: set_one does.
    std::pair<iterator, bool> _insert(const value_type& val) {
        assert(this->m_base);
        index_type mIdx = this->m_conv.m_from_v(val.first);

        // repackage the base iterator into the view iterator
        std::pair<typename Array::iterator, bool> res;
        res = this->m_base->_insert(std::make_pair(mIdx, val.second));
        iterator it = iterator(res.first, this->m_base->end(), this->m_conv);
        return std::make_pair(it, res.second);
    }

    data_type get_one(const index_type& vIdx) const {
        assert(this->m_base);
        index_type mIdx = this->m_conv.m_from_v(vIdx);
        return this->m_base->get_one(mIdx);
    }
    void set_one(const index_type& vIdx, const data_type& value) {
        assert(this->m_base);
        index_type mIdx = this->m_conv.m_from_v(vIdx);
        this->m_base->set_one(mIdx, value);
    }

    void set_one(const iterator& it, const data_type& value) {
        assert(this->m_base);
        it.base()->second = value;
    }

    virtual single_index_type _flat_index(const index_type& vIdx) const {
        return _flat_index_helper<data_type, single_index_type>(vIdx, shape(), ndim());
    }

};


} // end namespace sparray


template<typename T, typename I>
std::ostream&
operator<<(std::ostream& out, const sparray::abstract_view_t<T, I>& ma)
{
    out << "\n*** VIEW :: shape = " << ma.shape() << "  num_elem = "<< ma.count_nonzero() <<" {\n";
    typename sparray::abstract_view_t<T, I>::const_iterator it = ma.cbegin();
    for (; it != ma.cend(); ++it){
        std::cout << "    " << it->first << " -> " << it->second <<"\n";
    }
    return out << "}  w/ fill_value = " << ma.fill_value() << "\n";
}

/*
 * TODO:
 * 1. make slicelength computed? or verify
 * 2. index_type, ndim; maybe an index_factory belongs to a view? otherwise,
 *     typedef map_array_t<double, single_index_type> mtype;
 *     mtype m(ndim);
 *     mtype::index_type idx(ndim);   // have to repeat ndim
 * 3. unify int ndim & size_t ndim;
 */

#endif
