# A 'sparse array' wrapper

# distutils: language = c++

### TODO:
#  1. expose dimensionality template param
#  2. need to expose iterators to sp_map_t?
#  3. inplace operators (__iadd__ etc)
#
#  Minor quibbles:
#  1. access to elements of fixed_capacity: operator[] or ELEM macro
#  2. how to keep typedefs in sync between Cy and C++


cdef extern from "fixed_cap.h" namespace "sparray":
    cdef cppclass fixed_capacity[I]:
        fixed_capacity() except +
        fixed_capacity(const I*) except +
        I& operator[](size_t j) except +


ctypedef int single_index_type            # this needs ot be in sync w/C++
ctypedef fixed_capacity[int] index_type   # XXX: can't do fixed_capacity[single_index_type]


cdef extern from "sp_map.h" namespace "sparray":
    cdef cppclass map_array_t[T]:
        map_array_t() except +
        map_array_t(const map_array_t&) except +
        void copy_from_other(map_array_t&) except +

        size_t ndim() const
        index_type shape() const

        T fill_value() const
        void set_fill_value(T value)

        size_t count_nonzero() const

        T get_one(const index_type& idx) const 
        void set_one(const index_type& idx, const T& value)

        void inplace_unary_op(T (*fptr)(T, T, T), T, T)


cdef extern from "elementwise_ops.h" namespace "sparray":
    T linear_unary_op[T](T, T, T)
    T power_unary_op[T](T, T, T)


cdef class MapArray:
    cdef map_array_t[double] *thisptr

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

    # TODO: 1. implement/expose binary ops
    #        2. type casting 
    #        3. add more arithm ops

    def __iadd__(MapArray self, double other):
        # add a scalar
        self.thisptr.inplace_unary_op(linear_unary_op[double], 1., other)
        return self
        
    def __isub__(MapArray self, double other):
        # subtract a scalar
        self.thisptr.inplace_unary_op(linear_unary_op[double], 1., -other)
        return self

    def __add__(MapArray self, other):
        # TODO: type testing, needed?
        newobj = MapArray()
        newobj.thisptr.copy_from_other(self.thisptr[0])
        return newobj.__iadd__(other)
