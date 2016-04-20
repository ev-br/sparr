#
# This file is basically a shim with ctypedefs for common_types.h, so that
# these can be cimported.
#
cdef extern from "common_types.h" namespace "sparray":
    ctypedef long int single_index_type

    struct slice_t:
        single_index_type start, stop, step
        single_index_type slicelength
