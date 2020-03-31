cimport numpy as np
cimport cython
from definitions cimport *


cdef class EaaModel:
	cdef Py_ssize_t num_topics
	cdef Py_ssize_t num_covariates
	cdef Py_ssize_t num_docs
	cdef Py_ssize_t num_tokens
	cdef Py_ssize_t num_speakers
	cdef Py_ssize_t num_passes
	cdef object ldict
	cdef object lrec
	cdef object lsuppl
	cdef object lmodel
	cdef object dictionary
	cdef object document
	cdef dict doc2bow
	cdef dict covariates
	cdef dict doc2speaker
	cdef dict speaker2doc
	cdef dict token2bow
	cdef dict token2doc
	cdef dtype_t elbo_value
	cdef np.ndarray alpha
	cdef np.ndarray _zeta
	cdef np.ndarray _Omega
	cdef np.ndarray _inv_Omega
	cdef np.ndarray _mu
	cdef np.ndarray _Sigma
	cdef np.ndarray _phi
	cdef np.ndarray _xi
	cpdef inference(self, Py_ssize_t total_passes=?)
	cdef void do_estep(self)
	cdef void do_mstep(self)
	cdef void save_parameters(self)
	cdef dtype_t compute_elbo(self)
	cpdef print_topics(self, Py_ssize_t topics=?, Py_ssize_t topn=?, bint random=?)
	cpdef show_topics(self, Py_ssize_t topics=?, Py_ssize_t topn=?,
								bint log=?, bint formatted=?, bint random=?)
	cpdef show_topic(self, Py_ssize_t k, Py_ssize_t topn=?)
	cpdef print_topic(self, Py_ssize_t k, Py_ssize_t topn=?)
	cpdef report_estimates(self, Py_ssize_t sortby=?)
	cpdef report_tscores(self, Py_ssize_t sortby=?)
	cpdef print_docs(self, Py_ssize_t k, Py_ssize_t docs=?, Py_ssize_t topn=?)
	cpdef print_docinfo(self, Py_ssize_t d)
	cdef inline char* get_token(self, Py_ssize_t tokenid)
