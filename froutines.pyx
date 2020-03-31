#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np
cimport cython

from eaamodel cimport EaaModel

np.seterr(invalid='raise')


cpdef inline dtype_t func_f(dtype_t[:] beta_ik,
					Py_ssize_t i, Py_ssize_t k,
					EaaModel self, bint minus):
	"""Returns the value of f(beta_ik)."""
	cdef:
		Py_ssize_t d, h, h1, h2, l
		Py_ssize_t num_topics = self.num_topics
		Py_ssize_t num_covariates = self.num_covariates
		unsigned int[:] doc_ids = self.speaker2doc[i]
		dtype_t dotprod, f, v, vmv_prod
		dtype_t[:] x_d = np.empty(num_covariates)
		dtype_t[:] zeta_k = self._zeta[k, :]
		dtype_t[:] xi_k = self._xi[:, k]
		dtype_t[:, :] mu_i = self._mu[i, :]
		dtype_t[:, :] inv_Omega_k = self._inv_Omega[k, :]
		dtype_t[:, :, :] Sigma_i = self._Sigma[i, :]
		dict covariates = self.covariates
		mpfr_t exp_t, sumexp_t, opt_t, exp_k_t

	mpfr_init2(exp_t, 53)
	mpfr_init2(exp_k_t, 53)
	mpfr_init2(sumexp_t, 53)
	mpfr_init2(opt_t, 53)

	# Log Gaussian prior p(beta_{ik}|(zeta_k, Omega_k))
	vmv_prod = 0.0
	for h1 in xrange(num_covariates):
		v = 0.0
		for h2 in xrange(num_covariates):
			v += (beta_ik[h2] - zeta_k[h2]) * inv_Omega_k[h2, h1]
		vmv_prod += v * (beta_ik[h1] - zeta_k[h1])
	f = -0.5 * vmv_prod
	# \sum_{j=1}^{D_i} xi_{ijk} * \log \sigma(x_{itk} * beta_{ik})
	for d in doc_ids:
		x_d = covariates[d]
		#sumexp = 0.0
		mpfr_set_d(sumexp_t, 0.0, MPFR_RNDU)
		for l in xrange(num_topics):
			dotprod = 0.0
			if l == k:
				for h in xrange(num_covariates):
					dotprod += x_d[h] * beta_ik[h]
				#exp_k = gsl_sf_exp(dotprod)
				#sumexp += exp_k
				mpfr_set_d(opt_t, dotprod, MPFR_RNDU)
				mpfr_exp(exp_k_t, opt_t, MPFR_RNDU)
				mpfr_add(sumexp_t, sumexp_t, exp_k_t, MPFR_RNDU)
			else:
				for h in xrange(num_covariates):
					dotprod += x_d[h] * mu_i[l, h]
				vmv_prod = 0.0
				for h1 in xrange(num_covariates):
					v = 0.0
					for h2 in xrange(num_covariates):
						v += x_d[h2] * Sigma_i[l, h2, h1]
					vmv_prod += v * x_d[h1]
				#sumexp += gsl_sf_exp(dotprod + 0.5 * vmv_prod)
				mpfr_set_d(opt_t, dotprod + 0.5 * vmv_prod, MPFR_RNDU)
				mpfr_exp(exp_t, opt_t, MPFR_RNDU)
				mpfr_add(sumexp_t, sumexp_t, exp_t, MPFR_RNDU)
		#f += xi_k[d] * gsl_sf_log(exp_k / sumexp)
		mpfr_div(opt_t, exp_k_t, sumexp_t, MPFR_RNDU)
		mpfr_log(opt_t, opt_t, MPFR_RNDU)
		f += xi_k[d] * mpfr_get_d(opt_t, MPFR_RNDU)
	
	mpfr_clear(exp_t)
	mpfr_clear(exp_k_t)
	mpfr_clear(sumexp_t)
	mpfr_clear(opt_t)	

	if minus:
		return -f
	else:
		return f


cpdef inline np.ndarray[dtype_t, ndim=1] func_f_prime(dtype_t[:] beta_ik,
					Py_ssize_t i, Py_ssize_t k,
					EaaModel self, bint minus):
	"""Returns the value of f(beta_ik)."""
	cdef:
		Py_ssize_t d, h, h1, h2, l
		Py_ssize_t num_topics = self.num_topics
		Py_ssize_t num_covariates = self.num_covariates
		unsigned int[:] doc_ids = self.speaker2doc[i]
		dtype_t factor_dk, prob_dk, dotprod, vmv_prod, v
		dtype_t[:] f_prime = np.zeros(num_covariates)
		dtype_t[:] x_d = np.empty(num_covariates)
		dtype_t[:] zeta_k = self._zeta[k, :]
		dtype_t[:] xi_k = self._xi[:, k]
		dtype_t[:, :] mu_i = self._mu[i, :]
		dtype_t[:, :] inv_Omega_k = self._inv_Omega[k, :]
		dtype_t[:, :, :] Sigma_i = self._Sigma[i, :]
		dict covariates = self.covariates
		mpfr_t exp_t, sumexp_t, opt_t, exp_k_t

	mpfr_init2(exp_t, 53)
	mpfr_init2(exp_k_t, 53)
	mpfr_init2(sumexp_t, 53)
	mpfr_init2(opt_t, 53)

	# Log Gaussian prior p(beta_{ik}|(zeta_k, Omega_k))
	for h1 in xrange(num_covariates):
		for h2 in xrange(num_covariates):
			f_prime[h1] -= (beta_ik[h2] - zeta_k[h2]) * inv_Omega_k[h2, h1]

	#f_prime *= -1

	for d in doc_ids:
		x_d = covariates[d]
		# sumexp = 0.0
		mpfr_set_d(sumexp_t, 0.0, MPFR_RNDU)
		for l in xrange(num_topics):
			dotprod = 0.0
			if l == k:
				for h in xrange(num_covariates):
					dotprod += x_d[h] * beta_ik[h]
				#exp_k = gsl_sf_exp(dotprod)
				#sumexp += exp_k
				mpfr_set_d(opt_t, dotprod, MPFR_RNDU)
				mpfr_exp(exp_k_t, opt_t, MPFR_RNDU)
				mpfr_add(sumexp_t, sumexp_t, exp_k_t, MPFR_RNDU)
			else:
				for h in xrange(num_covariates):
					dotprod += x_d[h] * mu_i[l, h]
				vmv_prod = 0.0
				for h1 in xrange(num_covariates):
					v = 0.0
					for h2 in xrange(num_covariates):
						v += x_d[h2] * Sigma_i[l, h2, h1]
					vmv_prod += v * x_d[h1]
				#sumexp += gsl_sf_exp(dotprod + 0.5 * vmv_prod)
				mpfr_set_d(opt_t, dotprod + 0.5 * vmv_prod, MPFR_RNDU)
				mpfr_exp(exp_t, opt_t, MPFR_RNDU)
				mpfr_add(sumexp_t, sumexp_t, exp_t, MPFR_RNDU)
		mpfr_div(opt_t, exp_k_t, sumexp_t, MPFR_RNDU)
		prob_dk = mpfr_get_d(opt_t, MPFR_RNDU)
		#prob_dk = exp_k / sumexp
		factor_dk = xi_k[d] * (1 - prob_dk)
		for h in xrange(num_covariates):
			f_prime[h] += factor_dk * x_d[h]

	mpfr_clear(exp_t)
	mpfr_clear(exp_k_t)
	mpfr_clear(sumexp_t)
	mpfr_clear(opt_t)

	if minus:
		return -np.asarray(f_prime)
	else:
		return np.asarray(f_prime)


cpdef inline np.ndarray[dtype_t, ndim=2] func_f_hess(dtype_t[:] beta_ik,
					Py_ssize_t i, Py_ssize_t k,
					EaaModel self, bint minus):
	"""Returns the Hessian of f(beta_ik) at point x."""
	cdef:
		int s
		Py_ssize_t d, h, h1, h2, l
		Py_ssize_t num_topics = self.num_topics
		Py_ssize_t num_covariates = self.num_covariates
		unsigned int[:] doc_ids = self.speaker2doc[i]
		dtype_t factor_dk, dotprod, v, vmv_prod, prob_dk
		dtype_t[:] xi_k = self._xi[:, k]
		dtype_t[:] x_d = np.empty(num_covariates)
		dtype_t[:, :] m = -self._inv_Omega[k, :]
		dtype_t[:, :] mu_i = self._mu[i, :]
		dtype_t[:, :, :] Sigma_i = self._Sigma[i, :]
		dict covariates = self.covariates
		mpfr_t exp_t, sumexp_t, opt_t, exp_k_t

	mpfr_init2(exp_t, 53)
	mpfr_init2(exp_k_t, 53)
	mpfr_init2(sumexp_t, 53)
	mpfr_init2(opt_t, 53)

	for d in doc_ids:
		x_d = covariates[d]
		mpfr_set_d(sumexp_t, 0.0, MPFR_RNDU)
		for l in xrange(num_topics):
			dotprod = 0.0
			if l == k:
				for h in xrange(num_covariates):
					dotprod += x_d[h] * beta_ik[h]
				#exp_k = gsl_sf_exp(dotprod)
				#sumexp += exp_k
				mpfr_set_d(opt_t, dotprod, MPFR_RNDU)
				mpfr_exp(exp_k_t, opt_t, MPFR_RNDU)
				mpfr_add(sumexp_t, sumexp_t, exp_k_t, MPFR_RNDU)
			else:
				for h in xrange(num_covariates):
					dotprod += x_d[h] * mu_i[l, h]
				vmv_prod = 0.0
				for h1 in xrange(num_covariates):
					v = 0.0
					for h2 in xrange(num_covariates):
						v += x_d[h2] * Sigma_i[l, h2, h1]
					vmv_prod += v * x_d[h1]
				#sumexp += gsl_sf_exp(dotprod + 0.5 * vmv_prod)
				mpfr_set_d(opt_t, dotprod + 0.5 * vmv_prod, MPFR_RNDU)
				mpfr_exp(exp_t, opt_t, MPFR_RNDU)
				mpfr_add(sumexp_t, sumexp_t, exp_t, MPFR_RNDU)
		mpfr_div(opt_t, exp_k_t, sumexp_t, MPFR_RNDU)
		prob_dk = mpfr_get_d(opt_t, MPFR_RNDU)
		#prob_dk = exp_k / sumexp
		factor_dk = -xi_k[d] * (prob_dk - prob_dk**2)
		for h1 in xrange(num_covariates):
			for h2 in xrange(h1, num_covariates):
				m[h1, h2] += factor_dk * x_d[h1] * x_d[h2]
				if h1 != h2:
					m[h2, h1] = m[h1, h2]

	mpfr_clear(exp_t)
	mpfr_clear(exp_k_t)
	mpfr_clear(sumexp_t)
	mpfr_clear(opt_t)

	if minus:
		return -np.asarray(m)
	else:
		return np.asarray(m)


cpdef inline np.ndarray[dtype_t, ndim=2] func_f_hess_inv(dtype_t[:] beta_ik,
					Py_ssize_t i, Py_ssize_t k,
					EaaModel self, bint minus):
	"""Returns the inverse of the Hessian of f(beta_ik) at point x."""
	cdef:
		int s
		Py_ssize_t d, h, h1, h2, l
		Py_ssize_t num_topics = self.num_topics
		Py_ssize_t num_covariates = self.num_covariates
		unsigned int[:] doc_ids = self.speaker2doc[i]
		dtype_t factor_dk, dotprod, prob_dk, vmv_prod, v
		dtype_t[:] xi_k = self._xi[:, k]
		dtype_t[:] x_d = np.empty(num_covariates)		
		dtype_t[:, :] m = -self._inv_Omega[k, :]
		dtype_t[:, :] mu_i = self._mu[i, :]
		dtype_t[:, :, :] Sigma_i = self._Sigma[i, :]
		np.ndarray[dtype_t, ndim=2] result = np.empty((num_covariates, num_covariates))
		dict covariates = self.covariates
		gsl_matrix* f_hess = gsl_matrix_alloc(num_covariates, num_covariates)
		gsl_matrix* inverse = gsl_matrix_alloc(num_covariates, num_covariates)
		gsl_permutation* perm = gsl_permutation_alloc(num_covariates)
		mpfr_t exp_t, sumexp_t, opt_t, exp_k_t

	mpfr_init2(exp_t, 53)
	mpfr_init2(exp_k_t, 53)
	mpfr_init2(sumexp_t, 53)
	mpfr_init2(opt_t, 53)

	for d in doc_ids:
		x_d = covariates[d]
		mpfr_set_d(sumexp_t, 0.0, MPFR_RNDU)
		for l in xrange(num_topics):
			dotprod = 0.0
			if l == k:
				for h in xrange(num_covariates):
					dotprod += x_d[h] * beta_ik[h]
				#exp_k = gsl_sf_exp(dotprod)
				#sumexp += exp_k
				mpfr_set_d(opt_t, dotprod, MPFR_RNDU)
				mpfr_exp(exp_k_t, opt_t, MPFR_RNDU)
				mpfr_add(sumexp_t, sumexp_t, exp_k_t, MPFR_RNDU)
			else:
				for h in xrange(num_covariates):
					dotprod += x_d[h] * mu_i[l, h]
				vmv_prod = 0.0
				for h1 in xrange(num_covariates):
					v = 0.0
					for h2 in xrange(num_covariates):
						v += x_d[h2] * Sigma_i[l, h2, h1]
					vmv_prod += v * x_d[h1]
				#sumexp += gsl_sf_exp(dotprod + 0.5 * vmv_prod)
				mpfr_set_d(opt_t, dotprod + 0.5 * vmv_prod, MPFR_RNDU)
				mpfr_exp(exp_t, opt_t, MPFR_RNDU)
				mpfr_add(sumexp_t, sumexp_t, exp_t, MPFR_RNDU)
		mpfr_div(opt_t, exp_k_t, sumexp_t, MPFR_RNDU)
		prob_dk = mpfr_get_d(opt_t, MPFR_RNDU)
		#prob_dk = exp_k / sumexp
		factor_dk = -xi_k[d] * (prob_dk - prob_dk**2)
		for h1 in xrange(num_covariates):
			for h2 in xrange(h1, num_covariates):
				m[h1, h2] += factor_dk * x_d[h1] * x_d[h2]
				if h1 != h2:
					m[h2, h1] = m[h1, h2]

	for h1 in xrange(num_covariates):
		for h2 in xrange(h1, num_covariates):
			gsl_matrix_set(f_hess, h1, h2, m[h1, h2])
			if h1 != h2:
				gsl_matrix_set(f_hess, h2, h1, m[h2, h1])
	# Make LU decomposition of matrix f_hess
	gsl_linalg_LU_decomp(f_hess, perm, &s)
	# Invert matrix f_hess
	gsl_linalg_LU_invert(f_hess, perm, inverse)
	for h1 in xrange(num_covariates):
		for h2 in xrange(h1, num_covariates):
			result[h1, h2] = gsl_matrix_get(inverse, h1, h2)
			if h1 != h2:
				result[h2, h1] = result[h1, h2]	

	mpfr_clear(exp_t)
	mpfr_clear(exp_k_t)
	mpfr_clear(sumexp_t)
	mpfr_clear(opt_t)

	if minus:
		return -result
	else:
		return result

