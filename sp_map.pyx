# A 'sparse array' wrapper

# distutils: language = c++

### TODO:
#  1. expose dimensionality template param
#  2. need to expose iterators to sp_map_t?
#  3. inplace operators (__iadd__ etc)
#  4. check arg not None
#  5. translate C++ exceptions
#  6. construct from_dense
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
        void copy_from_other(const map_array_t*) except +

        size_t ndim() const
        index_type shape() const
        index_type get_min_shape() const
        void set_shape(const index_type&) except +

        T fill_value() const
        void set_fill_value(T value)

        size_t count_nonzero() const

        # single element accessors
        T get_one(const index_type& idx) const 
        void set_one(const index_type& idx, const T& value)

        void todense(void* dest, const single_index_type num_elem) except +

        void inplace_unary_op(T (*fptr)(T x, T a, T b), T a, T b)  # x <- f(x, a, b)
        void inplace_binary_op(T (*fptr)(T x, T y, T a, T b),
                               const map_array_t[T] *p_other, T a, T b) except +  # x <- f(x, y, a, b)


cdef extern from "elementwise_ops.h" namespace "sparray":
    T linear_unary_op[T](T, T, T)
    T power_unary_op[T](T, T, T)
    T linear_binary_op[T](T, T, T, T)


cdef index_type index_from_tuple(tuple tpl):
    # conversion helper: 2D only.
    cdef index_type idx
    idx[0] = tpl[0]
    idx[1] = tpl[1]
    return idx


<<<<<<< HEAD
=======
#ctypedef fused duck_t:
#    double
#    float
#    long


#cdef union union_t:
#    map_array_t[long] *longptr
#    map_array_t[float] *floatptr
#    map_array_t[double] *doubleptr


#cdef class MapArray2:
#    cdef union_t p
#    cdef object dtype

#    def __cinit__(self, dtype=float):

#        self.dtype = np.dtype(dtype)     # seems to convert float etc to numpy dtypes?

#        cdef int typenum = self.dtype.num
#        if typenum == cnp.NPY_DOUBLE: 
#            self.p.doubleptr = new map_array_t[double]()
#        elif typenum == cnp.NPY_FLOAT:
#            self.p.floatptr = new map_array_t[float]()
#        elif typenum == cnp.NPY_INT:
#            self.p.longptr = new map_array_t[long]()
#        else:
#            raise ValueError("dtype %s  not supported." % dtype)

#        print(">>> ", dtype, typenum)

#    def __dealloc__(self):
#        cdef int typenum = self.dtype.num

#        if typenum == cnp.NPY_DOUBLE: 
#            del self.p.doubleptr
#        elif typenum == cnp.NPY_FLOAT:
#            del self.p.floatptr
#        elif typenum == cnp.NPY_INT:
#            del self.p.longptr
#        else:
#            raise ValueError("Panic! Unsupported dtype %s  in dtor." % self.dtype)

#    property dtype:
#        def __get__(self):
#            return self.dtype

#    ###### Arithmetics #######

#    def __iadd__(self, other):
#        cdef int typenum = self.dtype.num

#        if typenum == cnp.NPY_DOUBLE: 
#            return iadd_method[double](self.p.doubleptr, other, 1.0)
#        elif typenum == cnp.NPY_FLOAT:
#            return iadd_method[float](self, other, 1.0)
#        else:
#            raise ValueError("Panic! Unsupported dtype %s  in __iadd__." % self.dtype)


#### Methods

#def iadd_method(self, other, duck_t unused):
#    cdef duck_t d_other
#    if isinstance(other, MapArray):
#        self.thisptr.inplace_binary_op(linear_binary_op[double],
#                                       other.thisptr, 1., 1.)
#        return self
#    elif isinstance(other, np.ndarray):
#        # hand over to __add__ for densification
#        return NotImplemented
#    else:
#        # it must be a scalar
#        try:
#            d_other = <duck_t?>(other)
#            self.thisptr.inplace_unary_op(linear_unary_op[duck_t], 1., d_other)
#            return self
#        except TypeError:
#            # strings, lists and other animals
#            return NotImplemented



#    cdef _iadd_maparr(MapArray self, MapArray other):
#        self.thisptr.inplace_binary_op(linear_binary_op[double],
#                                       other.thisptr, 1., 1.)
#        return self


###########################################################################


>>>>>>> MAINT: use tuple-> index-type helper, refactor __iadd__ store the typenum
cdef class MapArray:
    cdef map_array_t[double] *thisptr
    cdef cnp.NPY_TYPES typenum

    __array_priority__ = 10.1     # equal to that of sparse matrices

    def __init__(self, dtype=float, shape=None, fill_value=0):
        pass

    def __cinit__(self, dtype=float, shape=None, fill_value=0, *args, **kwds):
        self.thisptr = new map_array_t[double]()

        dtype = np.dtype(dtype)   # seems to convert float etc to numpy dtypes?
        self.typenum = dtype.num

        cdef index_type shp
        if shape:
            if len(shape) != self.thisptr.ndim():
                raise ValueError("Shape %s not undestood." % str(shape))

            shp = index_from_tuple(shape)
            self.thisptr.set_shape(shp)

        if fill_value is not None:
            self.thisptr.set_fill_value(fill_value)

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
        cdef index_type idx = index_from_tuple(tpl)
        return self.thisptr.get_one(idx)

    def __setitem__(self, tpl, value):
        cdef index_type idx = index_from_tuple(tpl)
        self.thisptr.set_one(idx, value)

    ###### Arithmetics #############

    def copy(self):
        newobj = MapArray()
        newobj.thisptr.copy_from_other(self.thisptr)
        return newobj

    # TODO: 
    #        2. type casting 
    #        3. add more arithm ops: sub, mul, div, l/r shifts, mod etc
    #        4. matmul and inplace axpy?

    def __iadd__(MapArray self not None, other):

        if isinstance(other, np.ndarray): 
            # hand over to __add__ for densification
            return NotImplemented

        if isinstance(other, MapArray):
            self.thisptr.inplace_binary_op(linear_binary_op[double],
                                           (<MapArray>other).thisptr, 1., 1.)
            return self

        # it must be a scalar then
        cdef double pod_other
        try:
            pod_other = <double?>(other)
            self.thisptr.inplace_unary_op(linear_unary_op[double], 1., pod_other)
            return self
        except TypeError:
            # strings, lists and other animals
            return NotImplemented

    def __add__(self, other):
        if isinstance(self, MapArray):
            if isinstance(other, np.ndarray):
                # Densify and return dense result
                return self.todense() + other
            else:
                newobj = self.copy()
                return newobj.__iadd__(other)
        elif isinstance(other, MapArray):
            return other.__add__(self)
        else:
            # how come?
            raise RuntimeError("__add__ : never be here ", self, other)


    # XXX: unify. Either both cast (<MapArray>self).thisptr,  or both dispatch onto a cdef function.
    cdef _iadd_maparr(MapArray self, MapArray other):
        self.thisptr.inplace_binary_op(linear_binary_op[double],
                                       other.thisptr, 1., 1.)
        return self


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
