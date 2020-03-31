cimport numpy as np
cimport cython
from definitions cimport *


cdef inline dtype_t[:, :] e_log_dirichlet(dtype_t[:, :] param) except *
cdef inline dtype_t joint_norm(dtype_t[:, :, :] v) except *