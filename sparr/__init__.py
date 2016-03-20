from __future__ import division, absolute_import, print_function

__all__ = ["MapArray"]

from ._sp_map import MapArray

from numpy.testing import Tester
test = Tester().test
