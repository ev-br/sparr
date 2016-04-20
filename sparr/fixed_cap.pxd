#
# This file is basically a shim with ctypedefs in fixed_cap.h, so that
# these can be cimported.
#
from common_types cimport single_index_type


cdef extern from "fixed_cap.h" namespace "sparray":
    cdef cppclass index_type:
        index_type()
        single_index_type& operator[](size_t j) except +

    # this creates index_type instances with fixed size
    cdef cppclass index_factory_t:
        index_factory_t(const int n)
        int ndim() const
        index_type get_new() except +

cdef extern from "fixed_cap.h":
    single_index_type[] FC_ELEMS(index_type) nogil

