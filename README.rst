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

Arithmetic operations are elementwise

>>> m2 = m * m + 1
>>> m2.to_coo()
(array([  1765.,  10202.]), (array([0, 1]), array([3, 2])))


Installation
------------

This isn't much to package really, so it's just very standard::

    $ python setup.py build_ext -i
    $ nosetests test_basic.py


Building requires ``cython`` and ``numpy``, testing needs ``nose``. 
