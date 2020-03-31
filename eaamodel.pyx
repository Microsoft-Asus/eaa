#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np
cimport cython

import cPickle as pickle
import logging, os
import interfaces
import itertools
import scipy, scipy.optimize
import time
import utils

from cython_gsl cimport *
from froutines import func_f, func_f_prime, func_f_hess
from froutines cimport func_f_hess_inv
from approx_routines cimport approx_e_log_sigma
from misc cimport e_log_dirichlet, joint_norm
from store_hdf5 import *

# RPy2 interface
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import numpy2ri
robjects.conversion.py2ri = numpy2ri # Enables automatic conversion of numpy arrays into R ones


cdef:
	# Declare constants from user_config
	Py_ssize_t MAX_ITER = 100 # The maximum number of iterations in the function f minimization
	Py_ssize_t MAX_PASSES = 3
	Py_ssize_t ITERS = 8 # The maximum number of iterations at the E step
	Py_ssize_t WPT = 18 # Number of most representative words per each topic
	Py_ssize_t DPT = 5 # Number of random documents per each topic for manual check
	dtype_t EPS = 1e-3 # The E step convergence criterion
	dtype_t EPS_0 = 1e-5 # The ELBO convergence criterion
	dtype_t TOL = 1e-6 # Optimization tolerance
	str DATAPATH = './data/'
	object logger = logging.getLogger('eaa.eaamodel')
	object MCMCpack = importr('MCMCpack')

np.random.seed(948449)
np.seterr(invalid='raise')
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


cdef class EaaModel:

	def __cinit__(self, dictionary_slink, record_slink, suppl_slink, model_slink,
				 new=True, **kwargs):
		"""Optional arguments may include the values of the hyperparameters, namely alpha."""
		cdef:
			Py_ssize_t d, k, w
			Py_ssize_t num_topics = Model.num_topics(model_slink)
			Py_ssize_t num_covariates = Supplementary.num_covariates(suppl_slink)
			Py_ssize_t num_docs = Dictionary.num_docs(dictionary_slink)
			Py_ssize_t num_tokens = Dictionary.num_tokens(dictionary_slink)
			Py_ssize_t num_speakers = Record.num_speakers(record_slink)
			np.ndarray[dtype_t, ndim=1] alpha

		logger.info('Initializing the EaaModel instance.')
		
		# Global model parameters
		self.num_topics = num_topics
		self.num_covariates = num_covariates
		self.num_docs = num_docs
		self.num_tokens = num_tokens
		self.num_speakers = num_speakers

		# Links to data tables
		self.ldict = dictionary_slink
		self.lrec = record_slink
		self.lsuppl = suppl_slink
		self.lmodel = model_slink

		# Keep data in physical memory
		self.dictionary = dictionary_slink().token2id.read()
		self.document = record_slink().document.read()

		# Random access raises the memory costs
		# but speeds up computation
		try:
			self.doc2bow = pickle.load(open(DATAPATH + 'processed/doc2bow.pickle', 'rb'))
			self.doc2speaker = pickle.load(open(DATAPATH + 'processed/doc2speaker.pickle', 'rb'))
			self.covariates = pickle.load(open(DATAPATH + 'processed/covariates.pickle', 'rb')) 
		except IOError:
			logger.info('Creating auxiliary mappings to speed up routine access to data.')
			self.doc2bow = {}
			self.doc2speaker = {}
			self.covariates = {}
			doc2bow = record_slink().doc2bow.read()
			document = record_slink().document.read()
			covariates = suppl_slink().covariates.read()
			for d in xrange(num_docs):
				self.doc2bow[d] = doc2bow[doc2bow['document_id'] == d][['token_id', 'token_freq']].view(('u4', (2,)))
				self.doc2speaker[d] = document[document['document_id'] == d]['speaker_id'][0]
				self.covariates[d] = covariates[covariates['document_id'] == d]['x'][0]
				if d % PROGRESS_CNT == 0:
					logger.info('PROGRESS: reading document #%i' % d)
			pickle.dump(self.doc2bow, open(DATAPATH + 'processed/doc2bow.pickle', 'wb'))
			pickle.dump(self.doc2speaker, open(DATAPATH + 'processed/doc2speaker.pickle', 'wb'))
			pickle.dump(self.covariates, open(DATAPATH + 'processed/covariates.pickle', 'wb'))
			del doc2bow, document, covariates # Free up some memory
		
		try:
			self.speaker2doc = pickle.load(open(DATAPATH + 'processed/speaker2doc.pickle', 'rb'))
		except IOError:
			logger.info('Creating auxiliary mappings to speed up routine access to data.')
			self.speaker2doc = {}
			document = record_slink().document.read()
			for i in xrange(num_speakers):
				self.speaker2doc[i] = document[document['speaker_id'] == i]['document_id']
			pickle.dump(self.speaker2doc, open(DATAPATH + 'processed/speaker2doc.pickle', 'wb'))
			del document

		try:
			self.token2doc = pickle.load(open(DATAPATH + 'processed/token2doc.pickle', 'rb'))
			self.token2bow = pickle.load(open(DATAPATH + 'processed/token2bow.pickle', 'rb'))
		except IOError:
			logger.info('Creating auxiliary mappings to speed up routine access to data.')
			self.token2doc = {}
			self.token2bow = {}
			doc2bow = record_slink().doc2bow.read()
			for w in xrange(num_tokens):
				self.token2doc[w] = doc2bow[doc2bow['token_id'] == w]['document_id']
				document_ids = self.token2doc[w]
				self.token2bow[w] = np.concatenate([self.doc2bow[d][self.doc2bow[d][:,0] == w] for d in document_ids])
			pickle.dump(self.token2doc, open(DATAPATH + 'processed/token2doc.pickle', 'wb'))
			pickle.dump(self.token2bow, open(DATAPATH + 'processed/token2bow.pickle', 'wb'))
			del doc2bow

		# Set priors
		alpha = kwargs.get('alpha')
		self.alpha = np.empty(num_tokens)
		if alpha is None:
			self.alpha[:] = 1. / num_tokens # Flat priors assumed
		else:
			self.alpha = alpha

		self.num_passes = Model.num_passes(self.lmodel)

		if new or (self.num_passes == 0):
			self.num_passes = 0
			Model.init(self.lmodel)
			# Initialize the higher-level parameters, zeta_k and Omega_k
			#self._zeta = np.random.multivariate_normal(np.zeros(num_covariates), 5*np.eye(num_covariates), (num_topics,))
			self._zeta = np.zeros((num_topics, num_covariates))
			self._Omega = np.empty((num_topics, num_covariates, num_covariates))
			self._inv_Omega = np.empty((num_topics, num_covariates, num_covariates))
			for k in xrange(num_topics):
				# Inverse Wishart as a prior
				self._Omega[k, :] = np.asarray(MCMCpack.riwish(num_covariates+4,
															(num_covariates+4)*np.eye(num_covariates)))
				self._inv_Omega[k, :] = scipy.linalg.inv(self._Omega[k, :])

			# Initialize the variational distribution q(theta_k|phi_k)
			self._phi = np.random.gamma(100., 1./100., (num_topics, num_tokens))
			
			# Initialize the variational distribution q(beta_{ik}|(mu_{ik}, Sigma_{ik}))
			self._mu = np.zeros((num_speakers, num_topics, num_covariates))
			self._Sigma = np.empty((num_speakers, num_topics, num_covariates, num_covariates))
			for i in xrange(num_speakers):
				for k in xrange(num_topics):
					self._Sigma[i, k, :] = np.eye(num_covariates)

			# Initialize the variational distribution q(z_d|xi_d)
			self._xi = np.random.dirichlet(np.ones(num_topics) / num_topics, num_docs)
			self.save_parameters()
			logger.info('Initialization is successful.')
		else:
			# Load data from existing tables
			logger.info('Continuing iterations from %d passes.' % self.num_passes)
			self._phi = QTheta.load_table(self.lmodel)
			(self._mu, self._Sigma) = QBeta.load_table(self.lmodel, shape=(num_speakers, num_topics, num_covariates))
			self._xi = QZ.load_table(self.lmodel)
			(self._zeta, self._Omega) = EBayes.load_table(self.lmodel)
			self._inv_Omega = np.empty((num_topics, num_covariates, num_covariates))
			for k in xrange(num_topics):
				self._inv_Omega[k, :] = scipy.linalg.inv(self._Omega[k, :])

		self.elbo_value = self.compute_elbo()


	cpdef inference(self, Py_ssize_t total_passes=MAX_PASSES):
		cdef:
			Py_ssize_t p
			Py_ssize_t current_passes = self.num_passes
			dtype_t relchange, L_new
			mpfr_t op_t, frac_t, diff_t

		mpfr_init2(op_t, 53)
		mpfr_init2(frac_t, 53)
		mpfr_init2(diff_t, 53)

		for p in xrange(current_passes, total_passes):
			logger.info('PASS NO. %d' % (self.num_passes+1))
			self.do_estep()
			self.do_mstep()
			L_new = self.compute_elbo()
			logger.info('After %d pass(es), the surrogate variational objective equals %f' % (p+1, L_new))
			mpfr_set_d(op_t, L_new, MPFR_RNDU)
			mpfr_set_d(frac_t, self.elbo_value, MPFR_RNDU)
			mpfr_sub(diff_t, op_t, frac_t, MPFR_RNDU)
			mpfr_div(frac_t, diff_t, frac_t, MPFR_RNDU)
			relchange = abs(mpfr_get_d(frac_t, MPFR_RNDU))
			if mpfr_get_d(diff_t, MPFR_RNDU) < 0:
				logger.warning('The surrogate objective decreases.')
			self.elbo_value = L_new
			ELBO.add_value(self.lmodel, L_new)
			Model.inc_passes(self.lmodel)
			self.num_passes += 1
			# Check convergence
			if relchange < EPS_0:
				logger.info('EBayes has converged successfully, relchange %f' % relchange)
				break
			elif p == (total_passes-1):
				logger.warning('EBayes step has not converged, relchange %f' % relchange)
			else:
				logger.info('Continuing EBayes, relchange %f' % relchange)

		mpfr_clear(op_t)
		mpfr_clear(frac_t)
		mpfr_clear(diff_t)


	cdef void do_estep(self) except *:
		"""Iterate until converged or the maximum number of iterations achieved."""
		cdef:
			Py_ssize_t d, i, k, t, w
			Py_ssize_t num_topics = self.num_topics
			Py_ssize_t num_covariates = self.num_covariates
			Py_ssize_t num_docs = self.num_docs
			Py_ssize_t num_speakers = self.num_speakers
			Py_ssize_t num_tokens = self.num_tokens
			Py_ssize_t num_passes = self.num_passes
			dtype_t relchange
			dtype_t[:] e_log_sigma = np.empty(num_topics)
			dtype_t[:] x0 = 1e-2 * np.ones(num_covariates)
			dtype_t[:, :, :] mu_old = np.empty((num_speakers, num_topics, num_covariates))
			np.ndarray[dtype_t, ndim=1] xi_k = np.empty(num_docs)
			np.ndarray[unsigned int, ndim=2] bows
			np.ndarray[dtype_t, ndim=2] e_log_theta = np.empty((num_topics, num_tokens))
			np.ndarray[dtype_t, ndim=3] beta = np.empty((num_speakers, num_topics, num_covariates))
			np.ndarray[dtype_t, ndim=4] beta_cov = np.empty((num_speakers, num_topics, num_covariates, num_covariates))
			dict doc2bow = self.doc2bow
			dict doc2speaker = self.doc2speaker
			dict token2doc = self.token2doc
			dict token2bow = self.token2bow
			mpfr_t exp_t, sumexp_t, frac_t, op_t

		mpfr_init2(exp_t, 53)
		mpfr_init2(op_t, 53)
		mpfr_init2(sumexp_t, 53)
		mpfr_init2(frac_t, 53)

		# Initialize the variational distribution q(beta_{ik}|(mu_{ik}, Sigma_{ik}))
		#self._mu = np.zeros((num_speakers, num_topics, num_covariates))
		self._mu = np.zeros((num_speakers, num_topics, num_covariates))
		mu_old = 1e-3 * np.ones((num_speakers, num_topics, num_covariates))
		self._Sigma = np.empty((num_speakers, num_topics, num_covariates, num_covariates))
		for i in xrange(num_speakers):
			for k in xrange(num_topics):
				self._Sigma[i, k, :] = np.eye(num_covariates)
		# Initialize the variational distribution q(theta_k|phi_k)
		self._phi = np.random.gamma(100., 1./100., (num_topics, num_tokens))
		# Initialize the variational distribution q(z_d|xi_d)
		self._xi = np.random.dirichlet(np.ones(num_topics) / num_topics, num_docs)

		for t in xrange(ITERS):
			# Update q(z_d|xi_d)
			logger.info('Updating q(z_d|xi_d)')
			e_log_theta = np.asarray(e_log_dirichlet(self._phi))
			for d in xrange(num_docs):
				bows = doc2bow[d]
				if not bows[:,0].any: # If the document contains extreme tokens only
					continue

				# Calculate the approximate value
				e_log_sigma[:] = approx_e_log_sigma(self, doc2speaker[d], d)
				
				# First calculate the normalizing constant, using the MPFR library to avoid overflow
				mpfr_set_d(sumexp_t, 0.0, MPFR_RNDU)
				for k in xrange(num_topics):
					mpfr_set_d(op_t, <double>(e_log_sigma[k] + np.dot(bows[:,1], e_log_theta[k,:][bows[:,0]])), MPFR_RNDU)
					mpfr_exp(exp_t, op_t, MPFR_RNDU)
					mpfr_add(sumexp_t, sumexp_t, exp_t, MPFR_RNDU)
				# Now calculate the 'actual' values
				for k in xrange(num_topics):
					mpfr_set_d(op_t, <double>(e_log_sigma[k] + np.dot(bows[:,1], e_log_theta[k,:][bows[:,0]])), MPFR_RNDU)
					mpfr_exp(exp_t, op_t, MPFR_RNDU)
					mpfr_div(frac_t, exp_t, sumexp_t, MPFR_RNDU)
					self._xi[d, k] = mpfr_get_d(frac_t, MPFR_RNDU)                               
				if d % PROGRESS_CNT == 0:
					logger.info('After %d pass(es) and %d iteration(s), xi_%d equals' % (num_passes+1, t+1, d))
					print self._xi[d,:]

			# Update q(beta_{ik}|(mu_{ik}, Sigma_{ik}))
			logger.info('Updating q(beta_{ik}|(mu_{ik}, Sigma_{ik}))')
			beta = np.empty((num_speakers, num_topics, num_covariates))
			beta_cov = np.empty((num_speakers, num_topics, num_covariates, num_covariates))
			for i in xrange(num_speakers):
				for k in xrange(num_topics):
					beta[i, k] = scipy.optimize.fmin_ncg(f=func_f,
															x0=x0,
															fprime=func_f_prime,
															fhess=func_f_hess,
															args=(i, k, self, True), maxiter=MAX_ITER,
															avextol=TOL, disp=False)
					beta_cov[i, k] = func_f_hess_inv(beta[i, k], i, k, self, True)
					if (i % int(num_speakers / 5) == 0) and (k == 0):
						logger.info('After %d pass(es) and %d iteration(s), mu_{%d,%d} equals' % (num_passes+1, t+1, i, k))
						print beta[i, k]
						logger.info('After %d pass(es) and %d iteration(s), Sigma_{%d,%d} equals' % (num_passes+1, t+1, i, k))
						print beta_cov[i, k]
			self._mu = beta.copy()
			self._Sigma = beta_cov.copy()

			# Update q(theta_k|phi_k)
			logger.info('Updating q(theta_k|phi_k)')
			for k in xrange(num_topics):
				self._phi[k] = self.alpha
				xi_k = self._xi[:, k]
				for w in xrange(num_tokens): # Ensure that there are no gaps in token_id ordering
					self._phi[k, w] += np.dot(xi_k[token2doc[w]], token2bow[w][:,1])
				self._phi[k] /= np.sum(self._phi[k])
				if k % 10 == 0:
					logger.info('After %d pass(es) and %d iteration(s), phi_%d equals' % (num_passes+1, t+1, k))
					print self._phi[k]

			# Print out some topics for debugging
			self.print_topics(10, random=True)

			# Check convergence
			mpfr_set_d(op_t, joint_norm(self._mu), MPFR_RNDU)
			mpfr_set_d(frac_t, joint_norm(mu_old), MPFR_RNDU)
			mpfr_sub(op_t, op_t, frac_t, MPFR_RNDU)
			mpfr_div(frac_t, op_t, frac_t, MPFR_RNDU)
			relchange = abs(mpfr_get_d(frac_t, MPFR_RNDU))
			if relchange < EPS:
				logger.info('E step has converged successfully, relchange %f' % relchange)
				break
			elif t == (ITERS-1):
				logger.warning('E step has not converged, relchange %f' % relchange)
			else:
				logger.info('Continuing E step, relchange %f' % relchange)
				mu_old = self._mu.copy()

		self.save_parameters()

		mpfr_clear(exp_t)
		mpfr_clear(op_t)
		mpfr_clear(sumexp_t)
		mpfr_clear(frac_t)


	cdef void do_mstep(self) except *:
		"""One-step update of zeta_k and Omega_k."""
		cdef:
			Py_ssize_t k
			Py_ssize_t num_covariates = self.num_covariates
			Py_ssize_t num_topics = self.num_topics
			Py_ssize_t num_speakers = self.num_speakers
			Py_ssize_t num_passes = self.num_passes
			dtype_t[:, :] cov_ik = np.empty((num_covariates, num_covariates))

		logger.info('Updating zeta_k and Omega_k')
		for k in xrange(num_topics):
			self._zeta[k, :] = np.mean(self._mu[:, k, :], axis=0)
			self._Omega[k, :] = np.mean(self._Sigma[:, k, :], axis=0)# + np.cov(self._mu[:, k, :], rowvar=0)
			self._inv_Omega[k, :] = scipy.linalg.inv(self._Omega[k, :])
			logger.info('After %d pass(es), zeta_%d equals' % (num_passes+1, k))
			print self._zeta[k, :]
			logger.info('After %d pass(es), Omega_%d equals' % (num_passes+1, k))
			print self._Omega[k, :]
		
		self.save_parameters()


	cdef void save_parameters(self) except *:
		"""Keep the newly updated estimates in the HDF5 database 
		at the end of each EM iteration."""
		QZ.update(self.lmodel, self._xi)
		QTheta.update(self.lmodel, self._phi)
		QBeta.update(self.lmodel, self._mu, self._Sigma)
		EBayes.update(self.lmodel, self._zeta, self._Omega)
		logger.info('Parameters have been saved successfully.')


	cdef dtype_t compute_elbo(self) except *:
		cdef:
			Py_ssize_t d, h, i, k, l, w
			Py_ssize_t num_topics = self.num_topics
			Py_ssize_t num_covariates = self.num_covariates
			Py_ssize_t num_speakers = self.num_speakers
			Py_ssize_t num_docs = self.num_docs
			Py_ssize_t num_tokens = self.num_tokens
			unsigned int[:] doc_ids
			dtype_t sum_alpha, enthropy, v, vmv_prod
			dtype_t L = 0.0
			dtype_t[:] alpha = self.alpha
			dtype_t[:] dotprod = np.empty(num_topics)
			dtype_t[:] phi_k = np.empty(num_tokens)
			dtype_t[:] x_d = np.empty(num_covariates)
			dtype_t[:] xi_d = np.empty(num_topics)
			dtype_t[:, :] mu_i = np.empty((num_topics, num_covariates))
			dtype_t[:, :, :] Sigma_i = np.empty((num_topics, num_covariates, num_covariates))
			np.ndarray[unsigned int, ndim=2] bows
			np.ndarray[dtype_t, ndim=1] zeta_k = np.empty(num_covariates)
			np.ndarray[dtype_t, ndim=2] e_log_theta = np.asarray(e_log_dirichlet(self._phi))
			np.ndarray[dtype_t, ndim=2] matrix_sum = np.empty((num_covariates, num_covariates))
			np.ndarray[dtype_t, ndim=2] mu_k = np.empty((num_speakers, num_covariates))
			np.ndarray[dtype_t, ndim=3] Sigma_k = np.empty((num_speakers, num_covariates, num_covariates))
			mpfr_t exp_t, exp_k_t, sumexp_t, op_t

		mpfr_init2(exp_t, 53)
		mpfr_init2(exp_k_t, 53)
		mpfr_init2(sumexp_t, 53)
		mpfr_init2(op_t, 53)

		# E[log p(Y|theta, z)]
		for d in xrange(num_docs):
			bows = self.doc2bow[d]
			if not bows[:,0].any: # If the document contains extreme tokens only
				continue
			xi_d = self._xi[d, :]
			for k in xrange(num_topics):
				L += xi_d[k] * np.dot(bows[:,1], e_log_theta[k,:][bows[:,0]])

		# E[log p(z|beta)]
		for i in xrange(num_speakers):
			mu_i = self._mu[i, :]
			Sigma_i = self._Sigma[i, :]
			doc_ids = self.speaker2doc[i]
			for d in doc_ids:
				xi_d = self._xi[d, :]
				x_d = self.covariates[d]
				mpfr_set_d(sumexp_t, 0.0, MPFR_RNDU)
				# First compute the sumexp
				dotprod = np.zeros(num_topics)
				for l in xrange(num_topics):
					for h in xrange(num_covariates):
						dotprod[l] += x_d[h] * mu_i[l, h]
					vmv_prod = 0.0
					for h1 in xrange(num_covariates):
						v = 0.0
						for h2 in xrange(num_covariates):
							v += x_d[h2] * Sigma_i[l, h2, h1]
						vmv_prod += v * x_d[h1]
					mpfr_set_d(op_t, dotprod[l] + 0.5 * vmv_prod, MPFR_RNDU)
					mpfr_exp(exp_t, op_t, MPFR_RNDU)
					mpfr_add(sumexp_t, sumexp_t, exp_t, MPFR_RNDU)
				# Now compute the contribution for each k
				for k in xrange(num_topics):
					mpfr_set_d(op_t, dotprod[k], MPFR_RNDU)
					mpfr_exp(exp_k_t, op_t, MPFR_RNDU) 
					mpfr_div(op_t, exp_k_t, sumexp_t, MPFR_RNDU)
					mpfr_log(op_t, op_t, MPFR_RNDU)
					L += xi_d[k] * mpfr_get_d(op_t, MPFR_RNDU)

		# E[log p(beta)]
		for k in xrange(num_topics):
			enthropy = num_covariates * num_speakers * gsl_sf_log(2*np.pi)
			enthropy += num_speakers * np.linalg.slogdet(self._Omega[k,:])[1] # slogdet returns a tuple
			matrix_sum = np.zeros((num_covariates, num_covariates))
			mu_k = self._mu[:, k, :]
			zeta_k = self._zeta[k, :]
			Sigma_k = self._Sigma[:, k, :]
			for i in xrange(num_speakers):
				matrix_sum += Sigma_k[i, :] + (mu_k[i, :] - zeta_k) * (mu_k[i, :] - zeta_k).T
			enthropy += np.trace(self._inv_Omega[k, :] * matrix_sum)
			L += -0.5 * enthropy

		# E[log p(theta)]
		for k in xrange(num_topics):
			L += gsl_sf_lngamma(np.sum(alpha))
			for w in xrange(num_tokens):
				L += -gsl_sf_lngamma(alpha[w]) + (alpha[w] - 1) * e_log_theta[k, w]

		# E[log q(z)]
		for d in xrange(num_docs):
			xi_d = self._xi[d, :]
			for k in xrange(num_topics):
				L -= xi_d[k] * gsl_sf_log(xi_d[k]) 

		# E[log q(beta)]
		for k in xrange(num_topics):
			enthropy = num_covariates * num_speakers * gsl_sf_log(2*np.pi) + num_speakers
			enthropy += sum([np.linalg.slogdet(self._Sigma[i, k, :])[1] for i in xrange(num_speakers)])
			L -= -0.5 * enthropy

		# E[log q(theta)]
		for k in xrange(num_topics):
			phi_k = self._phi[k,:]
			L -= gsl_sf_lngamma(np.sum(phi_k))
			for w in xrange(num_tokens):
				L -= -gsl_sf_lngamma(phi_k[w]) + (phi_k[w] - 1) * e_log_theta[k, w]

		mpfr_clear(exp_t)
		mpfr_clear(exp_k_t)
		mpfr_clear(sumexp_t)
		mpfr_clear(op_t)

		return L


	cpdef print_topics(self, Py_ssize_t topics=10, Py_ssize_t topn=WPT, bint random=False):
		self.show_topics(topics, topn, log=True, random=random)


	cpdef show_topics(self, Py_ssize_t topics=10, Py_ssize_t topn=10,
								bint log=False, bint formatted=True, bint random=False):
		"""
		Print the `topN` most probable words for (randomly selected) `topics`
		number of topics. Set `topics=-1` to print all topics.
		"""
		cdef:
			Py_ssize_t k
			Py_ssize_t num_topics = self.num_topics
		if topics < 0:
			# Print all topics if `topics` is negative
			topics = num_topics
		if random:
			ids = np.random.choice(num_topics, size=(min(topics, num_topics),), replace=False)
		else:
			ids = range(min(topics, num_topics))
		shown  = []
		for k in ids:
			if formatted:
				topic = self.print_topic(k, topn=topn)
			else:
				topic = self.show_topic(k, topn=topn)
			shown.append(topic)
			if log:
				print 'topic #%i: %s\n' % (k, topic)
		return shown


	cpdef show_topic(self, Py_ssize_t k, Py_ssize_t topn=10):
		topic = self._phi[k, :]
		bestn = np.argsort(topic)[::-1][:topn]
		beststr = [(topic[tokenid], self.get_token(tokenid)) for tokenid in bestn]
		return beststr


	cpdef print_topic(self, Py_ssize_t k, Py_ssize_t topn=10):
		return ' + '.join(['%.3f*%s' % v for v in self.show_topic(k, topn)])


	cpdef report_estimates(self, Py_ssize_t sortby=3):
		cdef:
			Py_ssize_t k
			Py_ssize_t num_topics = self.num_topics
			Py_ssize_t num_covariates = self.num_covariates
			Py_ssize_t[:] ordering = np.empty(num_topics, dtype=int)
			np.ndarray[dtype_t, ndim=2] t_scores = np.empty((num_topics, num_covariates))
		
		for k in xrange(num_topics):
			t_scores[k, :] = self._zeta[k, :] / np.sqrt(np.diag(self._Omega[k, :]))
		ordering = np.argsort(np.abs(t_scores[:, sortby]))[::-1]
		for k in ordering:
			print 'zeta_%d: %s\n' % (k, np.around(self._zeta[k, :], decimals=4))
		for k in ordering:
			print 'Omega_%d: \n%s\n' % (k, np.around(self._Omega[k, :], decimals=4))


	cpdef report_tscores(self, Py_ssize_t sortby=3):
		"""Returns 't-statistics'. Sortby must be the column index, from 0 to (H-1)."""
		cdef:
			Py_ssize_t k
			Py_ssize_t num_topics = self.num_topics
			Py_ssize_t num_covariates = self.num_covariates
			Py_ssize_t[:] ordering = np.empty(num_topics, dtype=int)
			np.ndarray[dtype_t, ndim=2] t_scores = np.empty((num_topics, num_covariates))
		
		for k in xrange(num_topics):
			t_scores[k, :] = self._zeta[k, :] / np.sqrt(np.diag(self._Omega[k, :]))
		ordering = np.argsort(np.abs(t_scores[:, sortby]))[::-1]
		for k in ordering:
			print 't-score_%d: %s\n' % (k, np.around(t_scores[k, :], decimals=4))


	cpdef print_docs(self, Py_ssize_t k, Py_ssize_t docs=DPT, Py_ssize_t topn=WPT):
		"Draw random documens from the given topic."
		cdef:
			Py_ssize_t d, n
			Py_ssize_t[:] ids
			np.ndarray[dtype_t, ndim=1] xi_k = self._xi[:, k]
		ids = np.argsort(xi_k)[::-1]
		n = len(ids)
		docs = min(docs, n)
		ids = np.random.choice(ids, size=(docs,), replace=False)
		print 'topic #%i: %s\n' % (k, self.print_topic(k, topn=topn))
		logger.info('%d out of %d randomly drawn documents for topic %d' % (docs, n, k))
		for d in ids:
			self.print_docinfo(d)
			print '%s\n' % [self.get_token(tokenid) for tokenid in self.doc2bow[d][:, 0]]


	cpdef print_docinfo(self, Py_ssize_t d):
		doc = self.document[d]
		print 'document #%d: fname %s, date %s' % (d, doc['fname'], doc['date'])


	cdef inline char* get_token(self, Py_ssize_t tokenid) except *:
		return self.dictionary[self.dictionary['token_id'] == tokenid]['token'][0]


	property num_docs:

		def __get__(self):
			return self.num_docs


	property num_topics:

		def __get__(self):
			return self.num_topics


	property num_covariates:

		def __get__(self):
			return self.num_covariates


	property num_tokens:

		def __get__(self):
			return self.num_tokens


	property num_speakers:

		def __get__(self):
			return self.num_speakers


	property num_passes:

		def __get__(self):
			return self.num_passes


	property covariates:

		def __get__(self):
			return self.covariates


	property _zeta:

		def __get__(self):
			return self._zeta

		def __set__(self, value):
			self._zeta = value


	property _Omega:

		def __get__(self):
			return self._Omega

		def __set__(self, value):
			self._Omega = value


	property _inv_Omega:

		def __get__(self):
			return self._inv_Omega

		def __set__(self, value):
			self._inv_Omega = value


	property _mu:

		def __get__(self):
			return self._mu

		def __set__(self, value):
			self._mu = value


	property _Sigma:

		def __get__(self):
			return self._Sigma

		def __set__(self, value):
			self._Sigma = value


	property _phi:

		def __get__(self):
			return self._phi


	property _xi:

		def __get__(self):
			return self._xi
#endclass EaaModel