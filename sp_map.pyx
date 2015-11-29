# A 'sparse array' wrapper

# distutils: language = c++

### TODO:
#  1. expose dimensionality template param
#  2. need to expose iterators to sp_map_t?
#  3. inplace operators (__iadd__ etc)
#  4. check arg not None
#  5. translate C++ exceptions
#
#  Minor quibbles:
#  1. access to elements of fixed_capacity: operator[] or ELEM macro
#  2. how to keep typedefs in sync between Cy and C++

import numpy as np
cimport numpy as cnp
from numpy cimport PyArray_SimpleNew, PyArray_DATA, PyArray_SIZE, npy_intp, NPY_DOUBLE


cdef extern from "fixed_cap.h" namespace "sparray":
    cdef cppclass fixed_capacity[I]:
        fixed_capacity() except +
        fixed_capacity(const I*) except +
        I& operator[](size_t j) except +
        long int[] elem_


ctypedef long int single_index_type            # this needs ot be in sync w/C++
ctypedef fixed_capacity[long int] index_type   # XXX: can't do fixed_capacity[single_index_type]


cdef extern from "sp_map.h" namespace "sparray":
    cdef cppclass map_array_t[T]:
        map_array_t() except +
        map_array_t(const map_array_t&) except +
        void copy_from_other(map_array_t&) except +

        size_t ndim() const
        index_type shape() const
        index_type get_min_shape() const

        T fill_value() const
        void set_fill_value(T value)

        size_t count_nonzero() const

        # single element accessors
        T get_one(const index_type& idx) const 
        void set_one(const index_type& idx, const T& value)

        void todense(void* dest, const single_index_type num_elem) const

        void inplace_unary_op(T (*fptr)(T x, T a, T b), T a, T b)  # x <- f(x, a, b)
        void inplace_binary_op(T (*fptr)(T x, T y, T a, T b),
                               const map_array_t[T]& other, T a, T b) except +  # x <- f(x, y, a, b)


cdef extern from "elementwise_ops.h" namespace "sparray":
    T linear_unary_op[T](T, T, T)
    T power_unary_op[T](T, T, T)
    T linear_binary_op[T](T, T, T, T)


cdef class MapArray:
    cdef map_array_t[double] *thisptr

    __array_priority__ = 10.1     # equal to that of sparse matrices

    def __cinit__(self):
        self.thisptr = new map_array_t[double]()

    def __dealloc__(self):
        del self.thisptr

    #### Public interface accessors #####

    property ndim:
        def __get__(self):
            return self.thisptr.ndim()

    property fill_value:
        def __get__(self):
            return self.thisptr.fill_value()

        def __set__(self, value):
            self.thisptr.set_fill_value(value)

    property shape:
        def __get__(self):
            cdef index_type sh = self.thisptr.shape()
            return sh[0], sh[1]   # TODO: ndim != 2

    def count_nonzero(self):
        return self.thisptr.count_nonzero()

    ##### Access single elements #####

    def __getitem__(self, tpl):
        # this is pretty much one big bug
        cdef int i = tpl[0], j = tpl[1]
        cdef index_type idx
        idx[0] = i
        idx[1] = j
        return self.thisptr.get_one(idx)

    def __setitem__(self, tpl, value):
        cdef index_type idx
        idx[0] = tpl[0]
        idx[1] = tpl[1]
        self.thisptr.set_one(idx, value)

    ###### Arithmetics #############

    def copy(self):
        newobj = MapArray()
        newobj.thisptr.copy_from_other(self.thisptr[0])
        return newobj

    # TODO: 
    #        2. type casting 
    #        3. add more arithm ops: sub, mul, div, l/r shifts, mod etc
    #        4. matmul and inplace axpy? 

    def __iadd__(self, other):
        cdef double d_other
        if isinstance(other, MapArray):
            return self._iadd_maparr(other)
        elif isinstance(other, np.ndarray):
            # hand over to __add__ for densification
            return NotImplemented
        else:
            # it must be a scalar
            try:
                d_other = <double?>(other)
                self.thisptr.inplace_unary_op(linear_unary_op[double], 1., d_other)
                return self
            except TypeError:
                # strings, lists and other animals
                return NotImplemented

    # XXX: unify. Either both cast (<MapArray>self).thisptr,  or both dispatch onto a cdef function.
    cdef _iadd_maparr(MapArray self, MapArray other):
        self.thisptr.inplace_binary_op(linear_binary_op[double],
                                       other.thisptr[0], 1., 1.)
        return self

    def __add__(self, other):
        if isinstance(self, MapArray):
            if isinstance(other, np.ndarray):
                # Densify return dense result
                return self.todense() + other
            else:
                newobj = MapArray()
                newobj.thisptr.copy_from_other((<MapArray>self).thisptr[0])  # XXX: copy_from_other(ptr)
                return newobj.__iadd__(other)
        elif isinstance(other, MapArray):
            return other.__add__(self)
        else:
            # how come?
            raise RuntimeError("__add__ : never be here ", self, other)


    def __isub__(MapArray self, double other):
        # subtract a scalar
        self.thisptr.inplace_unary_op(linear_unary_op[double], 1., -other)
        return self

    def todense(self):
        cdef int nd = <int>self.thisptr.ndim()
        cdef cnp.ndarray a = PyArray_SimpleNew(self.thisptr.ndim(),
                                               <npy_intp*>self.thisptr.shape().elem_,
                                               NPY_DOUBLE)
        self.thisptr.todense(PyArray_DATA(a), PyArray_SIZE(a))
        return a


cnp.import_array()
