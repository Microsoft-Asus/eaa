cimport numpy as np
cimport cython
from definitions cimport *

from eaamodel cimport EaaModel


cpdef inline dtype_t func_f(dtype_t[:] beta_ik,
					Py_ssize_t i, Py_ssize_t k,
					EaaModel self, bint minus)

cpdef inline np.ndarray[dtype_t, ndim=1] func_f_prime(dtype_t[:] beta_ik,
					Py_ssize_t i, Py_ssize_t k,
					EaaModel self, bint minus)

cpdef inline np.ndarray[dtype_t, ndim=2] func_f_hess(dtype_t[:] beta_ik,
					Py_ssize_t i, Py_ssize_t k,
					EaaModel self, bint minus)

cpdef inline np.ndarray[dtype_t, ndim=2] func_f_hess_inv(dtype_t[:] beta_ik,
					Py_ssize_t i, Py_ssize_t k,
					EaaModel self, bint minus)