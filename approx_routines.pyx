#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np
cimport cython

from cython_gsl cimport *
from eaamodel cimport EaaModel


cdef inline dtype_t[:] approx_e_log_sigma(EaaModel self, Py_ssize_t i, Py_ssize_t d) except *:
	"""Returns the approximate value of E[log sigma]."""
	cdef:
		Py_ssize_t h, h1, h2, k
		Py_ssize_t num_topics = self.num_topics
		Py_ssize_t num_covariates = self.num_covariates
		dtype_t v, vmv_prod
		dtype_t[:] dotprod = np.zeros(num_topics)
		dtype_t[:] x = self.covariates[d]
		dtype_t[:] mu_ik = np.empty(num_covariates)
		dtype_t[:] e_log_sigma = np.empty(num_topics)
		dtype_t[:, :, :] Sigma_i = self._Sigma[i, :]
		mpfr_t exp_t, sumexp_t, opt_t

	mpfr_init2(exp_t, 53)
	mpfr_init2(sumexp_t, 53)
	mpfr_init2(opt_t, 53)

	mpfr_set_d(sumexp_t, 0.0, MPFR_RNDU)
	for k in xrange(num_topics):
		mu_ik = self._mu[i, k, :]
		for h in xrange(num_covariates):
			dotprod[k] += x[h] * mu_ik[h]
		# Approximation using Jensen's inequality
		vmv_prod = 0.0
		for h1 in xrange(num_covariates):
			v = 0.0
			for h2 in xrange(num_covariates):
				v += x[h2] * Sigma_i[k, h2, h1]
			vmv_prod += v * x[h1]
		mpfr_set_d(opt_t, dotprod[k] + 0.5 * vmv_prod, MPFR_RNDU)
		mpfr_exp(exp_t, opt_t, MPFR_RNDU)
		mpfr_add(sumexp_t, sumexp_t, exp_t, MPFR_RNDU)
	for k in xrange(num_topics):
		mpfr_log(opt_t, sumexp_t, MPFR_RNDU)
		e_log_sigma[k] = dotprod[k] - mpfr_get_d(opt_t, MPFR_RNDU)

	mpfr_clear(exp_t)
	mpfr_clear(sumexp_t)
	mpfr_clear(opt_t)


	return e_log_sigma