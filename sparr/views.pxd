from libcpp.vector cimport vector
from common_types cimport slice_t, single_index_type
from fixed_cap cimport index_type

cdef extern from "sp_map.h" namespace "sparray":
    cdef cppclass map_array_t[T, I]:
        map_array_t(single_index_type ndim)
        void copy_from[S, J](const map_array_t[S, J]*) except +



#
# Views. Only expose the minimum needed functionality:
#   * for the base class, do not expose constructors
#   * for the concrete views, only expose constructors
#
# This way, the usage is to instantiate a concrete view and
# use its polymorphic methods via the base class pointer.
#
cdef extern from "views.h" namespace "sparray":

    cdef cppclass abstract_view_t[T, I]:
        single_index_type ndim() const
        index_type shape() const
        void set_shape(const index_type&) except +

        T fill_value() const
        void set_fill_value(T value)

        single_index_type count_nonzero() const

        # single element accessors
        T get_one(const index_type& idx) const 
        void set_one(const index_type& idx, const T& value)


    cdef cppclass map_view_t[T, I](abstract_view_t[T, I]):
        map_view_t(map_array_t[T, I] *base)


    cdef cppclass view_view_t[T, I](abstract_view_t[T, I]):
        view_view_t(abstract_view_t[T, I] *base_view,
                    const vector[slice_t]& slices)



#
# Operations.
#       XXX: here or operations.pxd?
#
cdef extern from "operations.h" namespace "sparray":
    void todense[T, I](abstract_view_t[T, I] *src,
                       void *dest,
                       const I num_elem) except +

    void to_coo[T, I](abstract_view_t[T, I] *src,
                      void* data, void* stacked_indices,
                      const I num_elem) except +

    void inplace_unary[T, I](abstract_view_t[T, I] *src,
                             T (*op)(T x),
                             T alpha, T beta)

    void inplace_binop[T, I](T (*binop)(T x, T y),
                             abstract_view_t[T, I] *self,
                             const abstract_view_t[T, I] *other) except +

    void apply_binop[S, T, I](T (*binop)(S x, S y),
                              abstract_view_t[T, I] *self,
                              const abstract_view_t[S, I] *arg1,
                              const abstract_view_t[S, I] *arg2) except +

    void apply_mmul[T, I](abstract_view_t[T, I] *C,
                          const abstract_view_t[T, I]* A,
                          const abstract_view_t[T, I]* B) except +
