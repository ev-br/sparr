# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

from sparr import MapArray as M

try:
    xrange
except NameError:
    # python 3
    xrange = range


def map_poisson2d(n):
    n2 = n*n
    L = M(shape=(n2, n2))
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

    params = [10, 100, 1000]

    def time_poisson2d(self, n):
        xxx = map_poisson2d(n)
    time_poisson2d.param_names = ['n']

    def peakmem_poisson2d(self, n):
        xxx = map_poisson2d(n)
    peakmem_poisson2d.param_names = ['n']


class GetSetPoisson2D(object):

    params = [10, 100, 1000]

    def setup(self, n):
        self.map_array = map_poisson2d(n)
        self.idx = self.map_array.shape[0] // 2

    def time_getitem_single(self, n):
        self.map_array[self.idx, -1]

    def time_setitem_single(self, n):
        self.map_array[self.idx, -1] = 101


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
