This is an experiment to try cooking up a sparse matrix class suitable for incremental assembly. It can be thought of as an alternative to ``scipy.sparse.dok_matrix``. Internally it wraps ``std::map``. 

While implementing a new sparse matrix class, it makes sense to make it a sparse *array*, so that ``a * b`` works
elementwise and ``a @ b`` is matrix multiplication.

Usage
-----

>>> from sparr import MapArray as M
>>> m = M()
>>> m.shape
(0, 0)

Inserting new elements expands the array

>>> m[1, 2] = -101
>>> m.shape
(2, 3)
>>> m[0, 3] = 42
>>> m.shape
(2, 4)
>>> m.todense()
array([[   0.,    0.,    0.,   42.],
       [   0.,    0., -101.,    0.]])

To have an array of fixed shape, set the `is_shape_fixed` attribute to ``True``,
or provide the shape explicitly in the constructor:

>>> m = M(shape=(2, 3))
>>> m.is_shape_fixed
True
>>> m[4, 5] = 6
Traceback (most recent call last)
...
IndexError: index 4 is out of bounds for axis 0 with size 2


The arrays have a `dtype` attribute and can be type-cast via `astype`:

>>> m.dtype
dtype('float64')
>>> i = m.astype(int)
>>> i.dtype
dtype('int64')

The rules for implicit type casting in arithmetic operations generally follow
numpy.

Arithmetic operations are elementwise

>>> m2 = m * m + 1
>>> m2.to_coo()
(array([  1765.,  10202.]), (array([0, 1]), array([3, 2])))

(Here the ``to_coo`` method returns the arrays ``data, (row, col)`` consistent
with `scipy.sparse.coo_matrix`. The reverse conversion is available via
the ``fom_coo`` method.)

Matrix multiplication operator is available on Python 3.5 and above:

>>> a = np.array([[0, -1], [2, 0]])
>>> m = M.from_dense(a)
>>> (m @ m).todense()
array([[-2, 0],
       [0, 2]])


Installation
------------

This isn't much to package really, so it's just very standard::

    $ python setup.py build_ext -i
    $ nosetests test_basic.py


Building requires ``cython`` and ``numpy``, testing needs ``nose``. 
