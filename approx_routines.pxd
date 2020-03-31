cimport numpy as np
cimport cython
from definitions cimport *

from eaamodel cimport EaaModel

cdef inline dtype_t[:] approx_e_log_sigma(EaaModel self, Py_ssize_t i, Py_ssize_t d) except *