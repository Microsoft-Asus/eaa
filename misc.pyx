#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport sqrt

cdef inline dtype_t[:, :] e_log_dirichlet(dtype_t[:, :] param) except *:
	"""For a vector r.v. ~ Dir(param), compute E[log(r.v.)]."""
	cdef:
		Py_ssize_t i, j
		Py_ssize_t size_i = param.shape[0]
		Py_ssize_t size_j = param.shape[1]
		dtype_t sum_axis
		dtype_t[:] param_i
		dtype_t[:, :] result = np.empty((size_i, size_j))
	
	for i in xrange(size_i):
		sum_axis = 0.0
		param_i = param[i, :]
		for j in xrange(size_j):
			sum_axis += param_i[j]
		for j in xrange(size_j):
			result[i, j] = gsl_sf_psi(param_i[j]) - gsl_sf_psi(sum_axis)
	return result


cdef inline dtype_t joint_norm(dtype_t[:, :, :] v) except *:
	cdef:
		Py_ssize_t i, j, k
		Py_ssize_t size_i = v.shape[0]
		Py_ssize_t size_j = v.shape[1]
		Py_ssize_t size_k = v.shape[2]
		dtype_t sum_axis
		dtype_t result = 0.0

	for i in xrange(size_i):
		for j in xrange(size_j):
			sum_axis = 0.0
			for k in xrange(size_k):
				sum_axis += v[i, j, k] ** 2
			result += sqrt(sum_axis)

	return result