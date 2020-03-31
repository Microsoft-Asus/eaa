#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from corpora.cngrec_corpus import CngrecCorpus
from corpora.cngrec_dictionary import CngrecDictionary
from user_config import *

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def process_cngrec():
	"""Initial processing."""
	corpus = CngrecCorpus(DATAPATH + 'corpora/')
	corpus.save(DATAPATH + 'processed/cngrec-preprocessed.pickle')


def filter_cngrec():
	"""Filtering extremes and serializing."""
	corpus = CngrecCorpus.load(DATAPATH + 'processed/cngrec-preprocessed.pickle')
	# Filter extremes
	id2word = corpus.dictionary
	id2word.save_as_text(DATAPATH + 'processed/cngrec-preprocessed.vocab')
	id2word.filter_extremes(
		no_below=NO_BELOW, no_above=NO_ABOVE, keep_n=VOCAB_SIZE)
	id2word.save_as_text(DATAPATH + 'processed/cngrec-preprocessed-filtered.vocab')
	corpus.save(DATAPATH + 'processed/cngrec-preprocessed-filtered.pickle')


def update():
	corpus = CngrecCorpus.load(DATAPATH + 'processed/cngrec-preprocessed-filtered.pickle')
	#print corpus.covariates['1366']


if __name__ == '__main__':	
	process_cngrec()
	#filter_cngrec()
	#update()