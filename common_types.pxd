#
# This file is basically a shim with ctypedefs for common_types.h, so that
# these can be cimported.
#
cdef extern from "common_types.h":
    ctypedef long int single_index_type
