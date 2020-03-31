cimport numpy as np
cimport cython

from cython_gsl cimport *

ctypedef np.float64_t dtype_t


cdef extern from "gmp.h":
	pass


cdef extern from "mpfr.h":
	ctypedef void* mpfr_t[1]
	ctypedef int mpfr_prec_t
	cdef enum mpfr_rnd_t:
		MPFR_RNDN
		MPFR_RNDZ
		MPFR_RNDU
		MPFR_RNDD
		MPFR_RNDA
		MPFR_RNDF
		MPFR_RNDNA
	void mpfr_init2(mpfr_t, mpfr_prec_t) nogil
	void mpfr_clear(mpfr_t) nogil
	void mpfr_set(mpfr_t rop, mpfr_t op, mpfr_rnd_t rnd) nogil
	void mpfr_set_d(mpfr_t, double, mpfr_rnd_t) nogil
	double mpfr_get_d(mpfr_t, mpfr_rnd_t) nogil
	void mpfr_div(mpfr_t, mpfr_t, mpfr_t, mpfr_rnd_t) nogil
	void mpfr_add(mpfr_t, mpfr_t, mpfr_t, mpfr_rnd_t) nogil
	void mpfr_sub(mpfr_t, mpfr_t, mpfr_t, mpfr_rnd_t) nogil
	void mpfr_d_sub(mpfr_t, double, mpfr_t, mpfr_rnd_t) nogil
	void mpfr_mul_d (mpfr_t, mpfr_t, double, mpfr_rnd_t) nogil
	void mpfr_sqr(mpfr_t, mpfr_t, mpfr_rnd_t) nogil
	void mpfr_exp(mpfr_t, mpfr_t, mpfr_rnd_t) nogil
	void mpfr_log(mpfr_t, mpfr_t, mpfr_rnd_t) nogil