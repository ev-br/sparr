# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

from sparr import MapArray as M
from scipy.sparse import dok_matrix

try:
    from pysparse.sparse import spmatrix
except:
    pass

try:
    xrange
except NameError:
    # python 3
    xrange = range


def map_poisson2d(n, func_name):
    n2 = n*n
    if func_name == "map_array":
        L = M(shape=(n2, n2))
    elif func_name == "dok_matrix":
        L = dok_matrix((n2, n2))
    elif func_name == "ll_mat":
        L = spmatrix.ll_mat(n2, n2)
    for i in xrange(n):
        for j in xrange(n):
            k = i + n*j
            L[k,k] = 4
            if i > 0:
               L[k,k-1] = -1
            if i < n-1:
               L[k,k+1] = -1
            if j > 0:
               L[k,k-n] = -1
            if j < n-1:
               L[k,k+n] = -1
    return L


class BenchPoisson2D(object):

    params = ([10, 100], ['map_array', 'dok_matrix', 'll_mat'])

    def time_poisson2d(self, n, func_name):
        xxx = map_poisson2d(n, func_name)
    time_poisson2d.param_names = ['n', 'class']
    time_poisson2d.timeout = 120.0

    def peakmem_poisson2d(self, n, func_name):
        xxx = map_poisson2d(n, func_name)
    peakmem_poisson2d.param_names = ['n', 'class']
    peakmem_poisson2d.timeout = 120.0

    def mem_poisson2d(self, n, func_name):
        xxx = map_poisson2d(n, func_name)
    mem_poisson2d.param_names = ['n', 'class']
    mem_poisson2d.timeout = 120.0

#class TimeSuite:
#    """
#    An example benchmark that times the performance of various kinds
#    of iterating over dictionaries in Python.
#    """
#    def setup(self):
#        self.d = {}
#        for x in range(500):
#            self.d[x] = None

#    def time_keys(self):
#        for key in self.d.keys():
#            pass

#    def time_iterkeys(self):
#        for key in self.d.iterkeys():
#            pass

#    def time_range(self):
#        d = self.d
#        for key in range(500):
#            x = d[key]

#    def time_xrange(self):
#        d = self.d
#        for key in xrange(500):
#            x = d[key]


#class MemSuite:
#    def mem_list(self):
#        return [0] * 256
