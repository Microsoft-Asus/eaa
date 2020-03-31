# -*- coding: utf-8 -*-
#
# This module contains implementation of the TextCorpus class 
# needed for preprocessing of the raw Congressional Record data.
# 
# This code builds on gensim.utils.

import logging, os
import pickle
import numpy
from cngrec_dictionary import CngrecDictionary
from gensim import utils
from gensim.corpora.textcorpus import TextCorpus
from stemming.porter2 import stem
from pattern.en import parse
from user_config import *

logger = logging.getLogger('eaa.corpora.cngrec_corpus')


def pickle(obj, fname, protocol=pickle.HIGHEST_PROTOCOL):
    """Pickle object `obj` to file `fname`."""
    with open(fname, 'wb') as fout: # 'b' for binary, needed on Windows
        pickle.dump(obj, fout, protocol=protocol)


def lemmatize(content):
    """
	Use the English lemmatizer from `pattern` to extract tokens in
	their base form=lemma, e.g. "are, is, being" -> "be" etc.
	This is a smarter version of stemming, taking word context into account.

	Only considers nouns, verbs, adjectives and adverbs by default (=all other lemmas are discarded).
    """
    content = u' '.join(utils.tokenize(content, lower=True, errors='ignore'))
    parsed = parse(content, lemmata=True, collapse=False)
    result = []
    for sentence in parsed:
        for token, tag, _, _, lemma in sentence:
            if 2 <= len(lemma) <= 15 and not lemma.startswith('_'):
                if utils.ALLOWED_TAGS.match(tag):
                    result.append(lemma.encode('utf8'))
    return result


def preprocess(content, stemming=True):
	"""
	Lemmatize and stem. No preprocessing should be done after this step.
	"""
	lemmatized = lemmatize(content)
	if stemming:
		return [stem(token) for token in lemmatized]
	else:
		return lemmatized


class CngrecCorpus(TextCorpus):
	"""Implementation of the TextCorpus class.
	The input parameter is assumed to be the corpora path.

	The getstream() and get_texts() methods are overriden."""

	def __init__(self, texts_input, covariates_input=None):
		if texts_input is None:
			raise ValueError('cannot instantiate the corpus without input')
		self.texts_input = texts_input
		self.dictionary = CngrecDictionary()
		self._speeches = {}
		self._serialize()

		self._covariates = {}
		self._covariates_attached = False

		# Optional
		self.covariates_input = covariates_input
		if not covariates_input is None:
			self._attach_covariates()


	def __iter__(self):
		if self._covariates_attached:
			for speaker in self._speeches:
				for speech in self._speeches[speaker]:
					yield self._speeches[speaker][speech], self._covariates[speaker][speech]
		else:
			for speaker in self._speeches:
				for speech in self._speeches[speaker]:
					yield self._speeches[speaker][speech]


	def save(self, output):
		logger.info("storing corpus to %s" % output)
		pickle(self, output)


	def _serialize(self, progress_cnt=PROGRESS_CNT):
		for speaker, speech, text in self._get_texts():
			self._speeches[speaker][speech] = self.dictionary.doc2bow(text, allow_update=True)
			docno += 1
			if docno % progress_cnt == 0:
				logger.info("PROGRESS: serializing document #%i" % docno)
		self.length = docno


	def _attach_covariates(self, progress_cnt=PROGRESS_CNT):
		if self.covariates_input is None:
			raise ValueError('cannot attach the covariates values without input')
		docno = 0
		for speaker, speech, _ in self._get_texts():
			self._covariates[speaker][speech] = numpy.zeros(NUM_COVARIATES,
				dtype={'names': COVARIATES_NAMES, 'formats': COVARIATES_FORMATS})
			for covariate in COVARIATES_NAMES:
				self._covariates[speaker][speech][covariate] = 1 # Shortcut for const
			docno += 1
			if docno % progress_cnt == 0:
				logger.info("PROGRESS: attaching covariates to document #%i" % docno)
		self._covariates_attached = True


	def _get_texts(self):
		for cngr in os.listdir(self.texts_input):
			if os.path.isdir(os.path.join(self.texts_input, cngr)):
				for speaker in os.listdir(self.texts_input + cngr):
					if speaker not in self._speeches.keys():
						self._speeches[speaker], self._covariates[speaker] = {}, {}
					if os.path.isdir(os.path.join(self.texts_input, cngr + '/' + speaker)):
						for speech in os.listdir(self.texts_input + cngr + '/' + speaker):
							if os.path.isfile(os.path.join(self.texts_input, 
							cngr + '/' + speaker + '/' + speech)):
								yield speaker, speech, preprocess(self._get_stream(self.texts_input
											 + cngr + '/' + speaker + '/' + speech))


	def _get_stream(self, file):
		return open(file, 'r').read()


	def speeches():
	    doc = "The speeches property."
	    def fget(self):
	        return self._speeches
	    return locals()
	speeches = property(**speeches())


	def covariates():
	    doc = "The covariates property."
	    def fget(self):
	        return self._covariates
	    def fset(self, value):
	        self._covariates = value
	    return locals()
	covariates = property(**covariates())


	def covariates_attached():
	    doc = "The covariates_attached property."
	    def fget(self):
	        return self._covariates_attached
	    return locals()
	covariates_attached = property(**covariates_attached())