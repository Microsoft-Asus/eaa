#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Imil Nurutdinov, GNU GPL 3.0
#
# The module gensim.models.ldamodel has provided inspiration for certain
# parts of the optimization routine.
#
# The auxiliary scripts, corpora/cngrec_corpus.py and corpora/cngrec_dictionary.py,
# build on gensim.utils and gensim.corpora.dictionary, respectively.

import tables

import numpy as np

from eaamodel import EaaModel
from store_hdf5 import *
from user_config import *


if __name__ == '__main__':
	db = tables.openFile(DATAPATH + 'processed/db.h5', 'r+')
	input_data = db.root.input_data
	if input_data.__contains__('ldict'):
		ldict = input_data.ldict
	else:
		ldict = db.createSoftLink(input_data, 'ldict', '/input_data/dictionary')
	if input_data.__contains__('lrec'):
		lrec = input_data.lrec
	else:
		lrec = db.createSoftLink(input_data, 'lrec', '/input_data/record')
	if input_data.__contains__('lsuppl'):
		lsuppl = input_data.lsuppl
	else:
		lsuppl = db.createSoftLink(input_data, 'lsuppl', '/input_data/supplementary')
	if input_data.__contains__('lsuppl'):
		lsuppl = input_data.lsuppl
	else:
		lsuppl = db.createSoftLink(input_data, 'lsuppl', '/input_data/supplementary')
	if db.root.__contains__('lmodel'):
		lmodel = db.root.lmodel
	else:
		lmodel = db.createSoftLink(db.root, 'lmodel', '/model')
	#lmodel()._v_attrs.num_passes = 4
	eaa = EaaModel(ldict, lrec, lsuppl, lmodel, new=False)
	#eaa.inference()
	#ELBO.print_rows(lmodel)
	#eaa.print_topics(-1, topn=50, random=False)
	#eaa.report_estimates()
	#eaa.report_tscores()
	eaa.print_docs(1, docs=20, topn=150)

	ldict.remove()
	lsuppl.remove()
	lrec.remove()
	lmodel.remove()
	db.close()