#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging, os
import interfaces
import itertools
import numpy as np
import simplejson as json
import tables
import utils

from corpora.cngrec_corpus import preprocess
from simplejson.decoder import JSONDecodeError
from user_config import *
from utils import date2sow

logger = logging.getLogger('eaa.store_hdf5')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)


class Dictionary(interfaces.DataContainer):
	"""A wrapper around the Dictionary group.

	It is an abstract class, with static variables only, so 
	do not try to create an instance of it.

	Uses soft links to tables to emulate the 'self' instance."""
	token2id = utils.Token2Id


	@staticmethod
	def doc2bow(slink, document, allow_update=False):
		result = utils.TemporaryTable(slink, utils.BoW, expectedrows=2000) # A rare speech document contains more than 2k words
		document = sorted(document)
		for word_norm, group in itertools.groupby(document):
			frequency = len(list(group)) # How many times does this word appear in the input document
			tokenid = Dictionary.get(slink, word_norm)
			if allow_update:
				if tokenid is None: # First time we see this token
					# new id = number of ids made so far; NOTE this assumes there are no gaps in the id sequence!
					tokenid = Dictionary.num_tokens(slink)
					Dictionary.extend(slink, tokenid, word_norm)
				else:
					Dictionary.update(slink, tokenid, word_norm)
			elif tokenid is None:
				continue # Ignore non-dictionary tokens
			result.set(tokenid=tokenid, frequency=frequency)

		# Return a copy of the Table instance; the current TemporaryTable will be deleted
		return result.data.copy(newname='buff', overwrite=True)


	@staticmethod
	def compactify(slink):
		"""Get rid of the gaps in the mapping."""
		temp_token2id = utils.TemporaryTable(slink, utils.Token2Id)
		tokenid = 0
		for row in slink().token2id:
			temp_token2id.set(word_norm=row['token'], tokenid=tokenid, frequency=row['token_freq'])
			tokenid += 1
		# Remove the existing nodes
		slink().token2id._f_remove()
		# Save a copy of the Table instance; the current TemporaryTable will be deleted
		temp_token2id.data.copy(slink(), 'token2id', title='The token2id table')
		slink().token2id.flush()


	@staticmethod
	def filter_extremes(slink, no_below=0.05, no_above=0.95, keep_n=1e5):
		no_below_abs = int(no_below * Dictionary.num_docs(slink)) # Convert fractional threshold to absolute threshold
		no_above_abs = int(no_above * Dictionary.num_docs(slink))
		# Determine which tokens to keep
		good_ids = [row['token_id'] for row in slink().token2id.where('(no_below_abs <= token_freq) & (token_freq <= no_above_abs)')]
		good_ids = sorted(good_ids, reverse=True)
		if not (keep_n is None):
			good_ids = good_ids[:keep_n]
		Dictionary.filter_tokens(slink, good_ids=good_ids)


	@staticmethod
	def filter_tokens(slink, bad_ids=None, good_ids=None):
		if bad_ids is not None:
			logger.info('Filtering bad ids in dictionary mapping')
			for bad_id in bad_ids:
				nrow = [row.nrow for row in slink().token2id.where('token_id == bad_id')][0]
				slink().token2id.removeRows(nrow, nrow+1)
		if good_ids is not None:
			logger.info('Keeping good ids in dictionary mapping')
			good_ids = sorted(good_ids)
			temp_token2id = utils.TemporaryTable(slink, utils.Token2Id)
			tokenid = 0 # Create new token_id to get rid of the gaps in the mapping
			for row in slink().token2id:
				if row['token_id'] in good_ids:
					temp_token2id.set(word_norm=row['token'], tokenid=tokenid, frequency=row['token_freq'])
					tokenid += 1
			# Remove the existing nodes
			slink().token2id._f_remove()
			# Save a copy of the Table instance; the current TemporaryTable will be deleted
			temp_token2id.data.copy(slink(), 'token2id', title='The token2id table')
		slink().token2id.flush()
		slink()._v_attrs.num_tokens = slink().token2id.nrows


	@staticmethod
	def save_as_text(slink, fname):
		logger.info('Saving dictionary mapping to %s' % fname)
		with open(fname, 'wb') as fout:
			for row in slink().token2id:
				fout.write('%s\t%i\t%i\n' % (row['token'], row['token_id'], row['token_freq']))


	@staticmethod
	def load_from_text(slink, fname):
		"""Load the dictionary mapping from the text document.

		Note that this method overwrites the existing dictionary."""
		logger.info('Loading the dictionary mapping from %s' % fname)
		temp_token2id = utils.TemporaryTable(slink, utils.Token2Id)
		tokenid = 0 # Create new token_id to get rid of the gaps in the mapping
		with open(fname, 'r') as vocab:
			for line in vocab:
				word_norm = line.split('\t')[0]
				frequency = line.split('\t')[2]
				temp_token2id.set(word_norm=word_norm, tokenid=tokenid, frequency=frequency)
				tokenid += 1
		# Remove the existing nodes
		slink().token2id._f_remove()
		# Save a copy of the Table instance; the current TemporaryTable will be deleted
		temp_token2id.data.copy(slink(), 'token2id', title='The token2id table')
		slink().token2id.flush()


	@staticmethod
	def update(slink, tokenid, word_norm, value=1):
		"""Updates the existing entry."""
		for row in slink().token2id.where('token == word_norm'):
			# Assign a new token_id OR increase the frequency parameter
			row['token_id'] = tokenid
			row['token_freq'] += value
			row.update()
		slink().token2id.flush()


	@staticmethod
	def extend(slink, tokenid, word_norm, value=1):
		"""Extends the dictionary."""
		slink().token2id.append([(word_norm, tokenid, value)])
		slink().token2id.flush()
		slink()._v_attrs.num_tokens += 1


	@staticmethod
	def get(slink, word_norm):
		"""Returns token_id by token."""
		found = [row['token_id'] for row in slink().token2id.where('token == word_norm')]
		if not found: return None # Check if empty
		else: return found[0]


	@staticmethod
	def get_token(slink, tokenid):
		"""Returns token by token_id."""
		found = [row['token'] for row in slink().token2id.where('token_id == tokenid')]
		if not found: return None # Check if empty
		else: return found[0]


	@staticmethod
	def num_tokens(slink):
		return slink()._v_attrs.num_tokens
	

	@staticmethod
	def num_docs(slink):
		return slink()._v_attrs.num_docs
#endclass Dictionary


class Record(interfaces.DataContainer):
	"""A wrapper around the Congressional Record group.

	It is an abstract class, with static variables only, so 
	do not try to create an instance of it.

	Uses soft links to tables to emulate the 'self' instance."""

	@staticmethod
	def num_speakers(slink):
		return slink()._v_attrs.num_speakers
#endclass Record


class Speaker(interfaces.TableWrapper):
	"""A wrapper around the Speaker table. 

	The key for this table is (icpsr_id, congress).

	It is an abstract class, with static variables and methods only, so 
	do not try to create an instance of it."""
	cols = np.dtype([
		('speaker_id', np.uint16),
		('icpsr_id', np.uint32),
		('congress', 'S3'), # 3-symbol code, e.g., '108'
		('chamber', 'S1'), # 'R' for representative, 'S' for senator
		('party', 'S1'), # 'D' for Democrat, 'R' for Republican
		('state', 'S2') # 2-symbol code, e.g., 'CA'
		])


	@staticmethod
	def read(filestream, congresses_xml):
		"""Read the data from corpora and congresses.xml."""
		from lxml import etree
		people = etree.parse(congresses_xml)
		for cngr in os.listdir(filestream):
			if os.path.isdir(os.path.join(filestream, cngr)):
				for speaker in os.listdir(filestream + cngr):
					if os.path.isdir(os.path.join(filestream, cngr + '/' + speaker)):
						person = people.xpath('//congress-history/people[@session=%s]/person[@icpsrid=%s]' % (cngr, speaker))[0]
						party = person.getchildren()[0].attrib['party']
						if party == 'Democrat': party = PARTY_D
						elif party == 'Republican': party = PARTY_R
						else: continue # Ingore the Independents
						role = person.getchildren()[0].attrib['type']
						if role == 'sen': role = CHAMBER_S
						else: role = CHAMBER_H
						state = person.getchildren()[0].attrib['state']
						yield int(person.attrib['icpsrid']), cngr, role, party, state


	@staticmethod
	def print_rows(record_slink):
		for row in record_slink().speaker:
			print row.fetch_all_fields()
#endclass Speaker


class Document(interfaces.TableWrapper):
	"""A wrapper around the Document table.

	It is an abstract class, with static variables and methods only, so 
	do not try to create an instance of it."""
	cols = np.dtype([
		('document_id', np.uint32),
		('speaker_id', np.uint16),
		('fname', 'S26'),
		('date', 'S16') # in the 'yyyy-mm-dd hh:mm' format
		])


	@staticmethod
	def read(filestream):
		"""Read the data from corpora."""
		for cngr in os.listdir(filestream):
			if os.path.isdir(os.path.join(filestream, cngr)):
				for speaker in os.listdir(filestream + cngr):
					if os.path.isdir(os.path.join(filestream, cngr + '/' + speaker)):
						for fname in os.listdir(filestream + cngr + '/' + speaker):
							if os.path.isfile(os.path.join(filestream, cngr + '/' + speaker + '/' + fname)):
								with open(filestream + cngr + '/' + speaker + '/' + fname, 'r') as speech:
									yield int(speaker), fname, preprocess(speech.read(), stemming=True)


	@staticmethod
	def print_rows(record_slink):
		for row in record_slink().document:
			print row.fetch_all_fields()


	@staticmethod
	def get_doc(record_slink, documentid):
		found = [row for row in record_slink().document.where('document_id == documentid')]
		if not found: return None # Check if empty
		else: return found[0]
#endclass Document


class Doc2BoW(interfaces.TableWrapper):
	"""A wrapper around the Doc2BoW table.

	It is an abstract class, with static variables only, so 
	do not try to create an instance of it.

	Also, it does not implement the read() method. Use Document.read() 
	instead."""
	cols = np.dtype([
		('document_id', np.uint32),
		('token_id', np.uint32),
		('token_freq', np.uint32)
		])

	@staticmethod
	def print_rows(record_slink):
		for row in record_slink().doc2bow:
			print row.fetch_all_fields()
#endclass Doc2BoW


class Supplementary(interfaces.DataContainer):
	"""A wrapper around the Supplementary Data group.

	It is an abstract class, with static variables only, so 
	do not try to create an instance of it.

	Uses soft links to tables to emulate the 'self' instance."""

	@staticmethod
	def num_covariates(slink):
		return slink()._v_attrs.num_covariates
#endclass Supplementary
		

class Covariates(interfaces.TableWrapper):
	"""A wrapper around the Covariates table. This part of the hierarchy 
	is likely to be changed depending on the research question.

	It is an abstract class, with static variables and methods only, so 
	do not try to create an instance of it."""
	cols = np.dtype([
		('document_id', np.uint32),
		('x', (np.float64, (NUM_COVARIATES,)))
		])
		# Covariates:
		# 0) const
		# 1) chamber -- 0 for representative, 1 for senator
		# 2) party -- 0 for Democrats, 1 for Republicans
		# 3) dw_nominate -- 1st dimension of the DW-NOMINATE score
		# 4) primary -- 1 if state of the world is primary election
		# 5) general -- 1 if state of the world is general election


	@staticmethod
	def read(congresses_xml, elections_json, **kwargs):
		"""Read the data from the Speaker table, congresses.xml, and elections.json."""
		record_slink = kwargs.get('record_slink')
		if record_slink is None:
			logger.error('Soft link to the Congressional Record group missing! Cannot run without it.')
			raise ValueError
		from lxml import etree
		parser = etree.parse(congresses_xml)
		people = etree.XPath('//congress-history/people[contains(@session, $congress)]/person[contains(@icpsrid, $icpsr_id)]')
		elections = json.load(open(elections_json), 'utf-8')['elections']
		for doc in record_slink().document:
			speakerid = doc['speaker_id']
			current_date = doc['date']
			cngr = str(utils.date2congress(current_date))
			found = [row for row in record_slink().speaker.where('(speaker_id == speakerid) & (congress == cngr)')]
			if not found:
				print doc['fname'], doc['date']
			
			speaker = found[0]
			chamber = speaker['chamber']
			senate = chamber == CHAMBER_S
			party = speaker['party']
			republican = party == PARTY_R
			# district = ...
			
			election = [e for e in elections if e['congress'] == cngr][0]['states'][0][speaker['state']]
			primary = election['primary'] # Primary and general elections hold simultaneously for chambers/parties
			general = election['general']

			sow = {} # State of the world
			sow['primary'] = 0
			sow['primary_runoff'] = 0
			sow['general'] = 0
			sow['none'] = 0 # No need to include the 'none' dummy as a covariate
			if senate:
				if election['senate_election']:
					if election['primary_runoff']:
						if election['primary_runoff'][CHAMBER_S]:
							primary_runoff = election['primary_runoff'][CHAMBER_S][party]
						else:
							primary_runoff = False
					else:
						primary_runoff = False
					if election['general_runoff']:
						if election['general_runoff'][CHAMBER_S]:
							general_runoff = election['general_runoff'][CHAMBER_S][party]
						else:
							general_runoff = False
					else:
						general_runoff = False
					sow[date2sow(current_date, primary, primary_runoff, general, general_runoff)] = 1
				else:
					sow['none'] = 1
			else:
				if election['primary_runoff']:
					if election['primary_runoff'][chamber]:
						primary_runoff = election['primary_runoff'][chamber][party]
					else:
						primary_runoff = False
				else:
					primary_runoff = False
				if election['general_runoff']:
					if election['general_runoff'][chamber]:
						general_runoff = election['general_runoff'][chamber][party]
					else:
						general_runoff = False
				else:
					general_runoff = False
				sow[date2sow(current_date, primary, primary_runoff, general, general_runoff)] = 1
			
			person = people(parser, congress=cngr, icpsr_id=speaker['icpsr_id'])[0]
			ideology = abs(float(person.attrib['dw1stdim']))
			yield doc['document_id'], 1.0, int(senate), int(republican), ideology,\
					sow['primary'], sow['general']#,\
					#ideology*sow['primary'], ideology*sow['general']

	@staticmethod
	def print_rows(suppl_slink):
		for row in suppl_slink().covariates:
			print row.fetch_all_fields()
#endclass Covariates


class Model(interfaces.DataContainer):
	"""A wrapper around the Model group.

	It is an abstract class, with static variables only, so 
	do not try to create an instance of it.

	Uses soft links to tables to emulate the 'self' instance."""


	@staticmethod
	def init(slink):
		slink().ass_cnvrg.elbo.removeRows(1, slink().ass_cnvrg.elbo.nrows)
		slink()._v_attrs.num_passes = 0


	@staticmethod
	def num_topics(slink):
		return slink()._v_attrs.num_topics
	

	@staticmethod
	def num_passes(slink):
		return slink()._v_attrs.num_passes


	@staticmethod
	def inc_passes(slink):
		slink()._v_attrs.num_passes += 1
#endclass Model


class QTheta(interfaces.TableWrapper):
	"""A wrapper around the QTheta table which contains the estimated topic
	distributions. q(theta) is parametrized by phi.

	It is an abstract class, with static variables and methods only, 
	so do not try to create an instance of it."""
	cols = None

	@staticmethod
	def load_table(slink):
		return slink().var_distr.q_theta.read()['phi']


	@staticmethod
	def update(slink, data):
		"""The data vector is meant to be in the appropriate 
		np.ndarray format."""
		slink().var_distr.q_theta.modifyRows(start=0, stop=data.shape[0],
											rows=[(k, data[k,:]) for k in xrange(data.shape[0])])
		slink().var_distr.q_theta.flush()


	@staticmethod
	def print_rows(slink):
		for row in slink().var_distr.q_theta:
			print row.fetch_all_fields()
#endclass QTheta


class QBeta(interfaces.TableWrapper):
	"""A wrapper around the QTheta table which contains the estimated coefficients
	in the covariates. q(beta) is parametrized by (mu, Sigma).

	It is an abstract class, with static variables and methods only, 
	so do not try to create an instance of it."""
	cols = None

	@staticmethod
	def load_table(slink, shape):
		return (slink().var_distr.q_beta.read()['mu'].reshape(shape),
				slink().var_distr.q_beta.read()['Sigma'].reshape(shape + (shape[-1],)))


	@staticmethod
	def update(slink, mu, Sigma):
		"""The data vector is meant to be in the appropriate
		np.ndarray format."""
		slink().var_distr.q_beta.modifyRows(start=0, stop=mu.shape[0]*mu.shape[1],
											rows=[(i, k, mu[i, k, :], Sigma[i, k, :])
											for i in xrange(mu.shape[0]) # Over speakers
											for k in xrange(mu.shape[1])]) # Over topics
		slink().var_distr.q_beta.flush()


	@staticmethod
	def print_rows(slink):
		for row in slink().var_distr.q_beta:
			print row.fetch_all_fields()
#endclass QZ


class QZ(interfaces.TableWrapper):
	"""A wrapper around the QZ table which contains the estimated topic
	probabilities for each document. q(z) is parametrized by xi.

	It is an abstract class, with static variables and methods only, 
	so do not try to create an instance of it."""
	cols = np.dtype([
		('document_id', np.uint32),
		('xi', (np.float64, (NUM_TOPICS,)))
		])

	@staticmethod
	def load_table(slink):
		return slink().var_distr.q_z.read()['xi']


	@staticmethod
	def update(slink, data):
		"""The data vector is meant to be in the appropriate
		np.ndarray format."""
		slink().var_distr.q_z.modifyRows(start=0, stop=data.shape[0],
										rows=[(d, data[d,:]) for d in xrange(data.shape[0])])
		slink().var_distr.q_z.flush()


	@staticmethod
	def print_rows(slink):
		for row in slink().var_distr.q_z:
			print row.fetch_all_fields()
#endclass QZ


class EBayes(interfaces.TableWrapper):
	"""A wrapper around the Empirical Bayes Estimates table.

	It is an abstract class, with static variables only, so 
	do not try to create an instance of it.

	Uses soft links to tables to emulate the 'self' instance."""
	cols = None

	@staticmethod
	def load_table(slink):
		return (slink().var_distr.e_bayes.read()['zeta'], slink().var_distr.e_bayes.read()['Omega'])


	@staticmethod
	def update(slink, zeta, Omega):
		"""The data vector is meant to be in the appropriate
		np.ndarray format."""
		slink().var_distr.e_bayes.modifyRows(start=0, stop=zeta.shape[0],
											rows=[(k, zeta[k, :], Omega[k, :]) for k in xrange(zeta.shape[0])])
		slink().var_distr.e_bayes.flush()


	@staticmethod
	def print_rows(slink):
		for row in slink().var_distr.e_bayes:
			print row.fetch_all_fields()
#endclass EBayes		


class ELBO(interfaces.TableWrapper):
	"""A wrapper around the ELBO talbe. Note that here ELBO is a shortcut 
	for the surrogate variational objective function.

	It is an abstract class, with static variables and methods only, 
	so do not try to create an instance of it."""
	cols = np.dtype([
		('pass_no', np.uint16),
		('value', np.float64)
		])


	@staticmethod
	def add_value(slink, value):
		"""Add the updated value of the variational objective."""
		pass_no = Model.num_passes(slink)
		row = slink().ass_cnvrg.elbo.row
		row['pass_no'] = pass_no
		row['value'] = value
		row.append()
		slink().ass_cnvrg.elbo.flush()


	@staticmethod
	def print_rows(slink):
		for row in slink().ass_cnvrg.elbo:
			print row.fetch_all_fields()
#endclass ELBO


def create_database():
	"""Specifies the input data only."""
	db = tables.openFile(DATAPATH + 'processed/db.h5', 'w')
	
	# /
	input_data = db.createGroup('/', 'input_data', 'The Input Data group')

	# /input_data
	dictionary = db.createGroup(input_data, 'dictionary', 'The Dictionary group')
	dictionary._v_attrs.num_docs = 0 # Number of documents processed
	dictionary._v_attrs.num_tokens = 0 # Number of token->id mappings in the dictionary
	record = db.createGroup(input_data, 'record', 'The Congressional Record group')
	record._v_attrs.num_speakers = 0 # Number of unique speakers (by icpsrID)
	supplementary = db.createGroup(input_data, 'supplementary', 'The Supplementary Data group')

	# /input_data/dictionary
	token2id = db.createTable(dictionary, 'token2id', Dictionary.token2id, 'The token2id table', expectedrows=1e5)
	# Add index only once the token2id table is completed
	#indexrows = token2id.cols.token_id.createCSIndex()

	# /input_data/record
	speaker = db.createTable(record, 'speaker', Speaker.cols, 'The Speaker table', expectedrows=2500)
	a = tables.UInt32Atom()
	document = db.createTable(record, 'document', Document.cols, 'The Document table', expectedrows=5e4)
	doc2bow = db.createTable(record, 'doc2bow', Doc2BoW.cols, 'The Doc2BoW table', expectedrows=5e4)

	# /input_data/supplementary
	covariates = db.createTable(supplementary, 'covariates', Covariates.cols, 'The Covariates table', expectedrows=5e4)

	db.close()


def init_dictionary(fname=None, progress_cnt=PROGRESS_CNT):
	"""Initialize the dictionary."""
	db = tables.openFile(DATAPATH + 'processed/db.h5', 'a')
	input_data = db.root.input_data
	if input_data.__contains__('ldict'):
		ldict = input_data.ldict
	else:
		ldict = db.createSoftLink(input_data, 'ldict', '/input_data/dictionary')

	if fname is None:
		logger.info('No dictionary file supplied, will generate from corpora')
		d = 0
		for speaker_id, fname, speech in Document.read(DATAPATH + 'corpora/'):
			#print speech
			if d % progress_cnt == 0:
				logger.info('PROGRESS: serializing document #%i' % d)
			buff = Dictionary.doc2bow(ldict, speech, True)
			buff._f_remove()
			d += 1
		ldict()._v_attrs.num_docs = d
		Dictionary.save_as_text(ldict, DATAPATH + 'processed/db.vocab')
	else:
		logger.info('Using the pre-defined dictionary mapping and parameters')
		Dictionary.load_from_text(ldict, fname)
		ldict()._v_attrs.num_docs = NUM_DOCS
	
	token2id = input_data.dictionary.token2id
	ldict()._v_attrs.num_tokens = token2id.nrows
	#indexrows = token2id.cols.token_id.reIndex()
	indexrows = token2id.cols.token_id.createCSIndex()

	ldict.remove()
	db.close()


def filter_dictionary():
	"""Filter dictionary."""
	db = tables.openFile(DATAPATH + 'processed/db.h5', 'a')
	input_data = db.root.input_data
	if input_data.__contains__('ldict'):
		ldict = input_data.ldict
	else:
		ldict = db.createSoftLink(input_data, 'ldict', '/input_data/dictionary')

	#Dictionary.filter_extremes(ldict, NO_BELOW, NO_ABOVE, VOCAB_SIZE)
	#bad_ids = [15, 31, 32, 38, 40, 48, 50, 57, 80, 84, 88, 117, 122, 136, 143, 173, 186, 226, 292, 314, 406, 487]
	#bad_ids = [38]
	Dictionary.filter_tokens(ldict, bad_ids=bad_ids)
	Dictionary.compactify(ldict)
	Dictionary.save_as_text(ldict, DATAPATH + 'processed/db-filtered.vocab')
	
	token2id = input_data.dictionary.token2id
	ldict()._v_attrs.num_tokens = token2id.nrows
	indexrows = token2id.cols.token_id.reIndex()

	ldict.remove()
	db.close()


def read_speaker_data(overwrite=True):
	"""Fill in the Speaker table overwritting existing values (by default)."""
	db = tables.openFile(DATAPATH + 'processed/db.h5', 'a')
	record = db.root.input_data.record
	if overwrite:
		if record.__contains__('speaker'):
			# Delete existing node
			record.speaker._f_remove()
			speaker = db.createTable(record, 'speaker', Speaker.cols, 'The Speaker table', expectedrows=2500)
			indexrows = speaker.cols.speaker_id.createCSIndex()
			indexrows = speaker.cols.icpsr_id.createCSIndex()	
	# Fill in the Speaker table
	table = record.speaker
	speaker = table.row
	# Need to decode icpsrID into ordinary integers
	icpsr2int = {}
	for icpsrid, congress, chamber, party, state in Speaker.read(DATAPATH + 'corpora/', DATAPATH + '108-111_congresses.xml'):
		if not (icpsrid in icpsr2int.keys()):
			icpsr2int[icpsrid] = len(icpsr2int)
		speaker['speaker_id'] = icpsr2int[icpsrid]
		speaker['icpsr_id'] = icpsrid
		speaker['congress'] = congress
		speaker['chamber'] = chamber
		speaker['party'] = party
		speaker['state'] = state
		speaker.append()
	table.flush()

	record._v_attrs.num_speakers = len(icpsr2int)

	db.close()


def read_document_data(overwrite=True, progress_cnt=PROGRESS_CNT, verbose=False):
	"""Fill in the Document and Doc2BoW tables overwritting existing values (by default).

	Proceed this AFTER the dictionary was initialized."""
	db = tables.openFile(DATAPATH + 'processed/db.h5', 'a')
	input_data = db.root.input_data
	record = input_data.record
	if overwrite:
		# Delete existing nodes
		if record.__contains__('document'):
			record.document._f_remove()
			document = db.createTable(record, 'document', Document.cols, 'The Document table', expectedrows=5e4)
			indexrows = document.cols.document_id.createCSIndex()
			indexrows = document.cols.speaker_id.createCSIndex()
		if record.__contains__('doc2bow'):
			record.doc2bow._f_remove()
			doc2bow = db.createTable(record, 'doc2bow', Doc2BoW.cols, 'The Doc2BoW table', expectedrows=5e4)
			indexrows = doc2bow.cols.document_id.createCSIndex()
			indexrows = doc2bow.cols.token_id.createCSIndex()
	if input_data.__contains__('ldict'):
		ldict = input_data.ldict
	else:
		ldict = db.createSoftLink(input_data, 'ldict', '/input_data/dictionary')
	if ldict().__contains__('temp_0'):
		ldict().temp_0._f_remove()

	document_table = record.document
	doc2bow_table = record.doc2bow
	doc = document_table.row
	bow = doc2bow_table.row
	d = 0
	for icpsrid, fname, speech in Document.read(DATAPATH + 'corpora/'):
		doc['document_id'] = d
		doc['fname'] = fname
		temp = fname[-16:]
		date = temp[:10] + ' ' + temp[11:]
		cngr = str(utils.date2congress(date))
		# Check if speaker is 'valid', i.e., there is a reference to her in the Speaker table
		found = [row for row in record.speaker.where('(icpsr_id == icpsrid) & (congress == cngr)')]
		if not found:
			if verbose:
				logger.warn('Speaker %i (%sth Congress) is an orphan.' % (icpsrid, cngr))
			continue
		doc['date'] = date
		doc['speaker_id'] = found[0]['speaker_id']
		if d % progress_cnt == 0:
			logger.info('PROGRESS: serializing document #%i' % d)
		buff = Dictionary.doc2bow(ldict, speech, False)
		for row in buff:
			bow['document_id'] = d
			bow['token_id'] = row['token_id']
			bow['token_freq'] = row['token_freq']
			bow.append()
		doc.append()
		d += 1
	buff._f_remove()
	document_table.flush()
	doc2bow_table.flush()

	# Update the number of documents
	ldict()._v_attrs.num_docs = d
	ldict.remove()
	db.close()


def read_covariates_data(overwrite=True):
	"""Fill in the Covariates table overwritting existing values (by default)."""
	db = tables.openFile(DATAPATH + 'processed/db.h5', 'a')
	input_data = db.root.input_data
	record = input_data.record
	supplementary = input_data.supplementary
	if overwrite and supplementary.__contains__('covariates'):
		# Delete existing node
		supplementary.covariates._f_remove()
		covariates = db.createTable(supplementary, 'covariates', Covariates.cols, 'The Covariates table', expectedrows=5e4)
		supplementary._v_attrs.num_covariates = Covariates.cols['x'].shape[0]
		indexrows = covariates.cols.document_id.createCSIndex()
	if input_data.__contains__('lrec'):
		lrec = input_data.lrec
	else:
		lrec = db.createSoftLink(input_data, 'lrec', '/input_data/record')

	covar = covariates.row
	logging.info('Attaching covariates')
	for document_id, const, chamber, party, dw_nominate,\
		primary, general in Covariates.read(
			DATAPATH + '108-111_congresses.xml',
			DATAPATH + 'elections-108-111.json',
			record_slink=lrec):
		covar['document_id'] = document_id
		covar['x'] = [const, chamber, party, dw_nominate, primary, general]
		covar.append()
	covariates.flush()

	lrec.remove()
	db.close()


def add_model_description(num_topics=NUM_TOPICS):
	"""Adds the EAA model description overwritting existing tables.
	Also, fills in the default values so that only editing data available then."""
	db = tables.openFile(DATAPATH + 'processed/db.h5', 'a')
	if db.root.__contains__('model'):
		for node in db.root.model:
			node._f_remove(recursive=True)
		db.root.model._f_remove()
	if db.root.__contains__('lmodel'):
		db.root.lmodel.remove()
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
	num_covariates = Supplementary.num_covariates(lsuppl)

	# /
	model = db.createGroup('/', 'model', 'The Model group')
	lmodel = db.createSoftLink('/', 'lmodel', '/model')
	model._v_attrs.num_topics = num_topics # Set the number of topics
	model._v_attrs.num_passes = 0 # The total number of passes in Variational EM

	# /model
	var_distr = db.createGroup(model, 'var_distr', 'The Variational Distribution group')
	ass_cnvrg = db.createGroup(model, 'ass_cnvrg', 'The Assessing Convergence group')

	# /model/var_distr
	QTheta.cols = np.dtype([
		('topic_id', np.uint16),
		('phi', (np.float64, (Dictionary.num_tokens(ldict),)))
		])
	q_theta = db.createTable(var_distr, 'q_theta', QTheta.cols, 'The q(theta) table', expectedrows=2.5e4)
	indexrows = q_theta.cols.topic_id.createCSIndex()
	default_row = q_theta.row
	for k in xrange(num_topics):
		default_row['topic_id'] = k
		default_row['phi'] = np.zeros(Dictionary.num_tokens(ldict), dtype=np.float64)
		default_row.append()
	q_theta.flush()

	QBeta.cols = np.dtype([
		('speaker_id', np.uint32),
		('topic_id', np.uint16),
		('mu', (np.float64, (num_covariates,))),
		('Sigma', (np.float64, (num_covariates, num_covariates)))
		])
	q_beta = db.createTable(var_distr, 'q_beta', QBeta.cols, 'The q(beta) table', expectedrows=500)
	indexrows = q_beta.cols.speaker_id.createCSIndex()
	indexrows = q_beta.cols.topic_id.createCSIndex()
	default_row = q_beta.row
	for i in xrange(Record.num_speakers(lrec)):
		for k in xrange(num_topics):
			default_row['speaker_id'] = i
			default_row['topic_id'] = k
			default_row['mu'] = np.zeros(num_covariates, dtype=np.float64)
			default_row['Sigma'] = np.zeros((num_covariates, num_covariates),
											dtype=np.float64)
			default_row.append()
	q_beta.flush()

	q_z = db.createTable(var_distr, 'q_z', QZ.cols, 'The q(z) table', expectedrows=2.5e5)
	indexrows = q_z.cols.document_id.createCSIndex()
	default_row = q_z.row
	for d in xrange(Dictionary.num_docs(ldict)):
		default_row['document_id'] = d
		default_row['xi'] = np.zeros(num_topics, dtype=np.float64)
		default_row.append()
	q_z.flush()

	EBayes.cols = np.dtype([
		('topic_id', np.uint16),
		('zeta', (np.float64, (num_covariates,))),
		('Omega', (np.float64, (num_covariates, num_covariates)))
		])
	e_bayes = db.createTable(var_distr, 'e_bayes', EBayes.cols, 'The Empirical Bayes Estimates table', expectedrows=100)
	indexrows = e_bayes.cols.topic_id.createCSIndex()
	default_row = e_bayes.row
	for k in xrange(num_topics):
		default_row['topic_id'] = k
		default_row['zeta'] = np.zeros(num_covariates, dtype=np.float64)
		default_row['Omega'] = np.zeros((num_covariates, num_covariates),
										dtype=np.float64)
		default_row.append()
	e_bayes.flush()

	# /model/ass_cnvrg
	elbo = db.createTable(ass_cnvrg, 'elbo', ELBO.cols, 'The ELBO table', expectedrows=100)
	row = elbo.row
	# A fictitious record
	row['pass_no'] = 0
	row['value'] = 0
	row.append() 
	elbo.flush()

	ldict.remove()
	lrec.remove()
	lsuppl.remove()
	lmodel.remove()
	db.close()


def show_info():
	"""Prints out the database description and removes temporary soft links."""
	db = tables.openFile(DATAPATH + 'processed/db.h5', 'a')
	input_data = db.root.input_data

	if input_data.__contains__('ldict'):
		ldict = input_data.ldict
	else:
		ldict = db.createSoftLink(input_data, 'ldict', '/input_data/dictionary')

	if ldict().__contains__('temp_0'):
		ldict().temp_0._f_remove()

	if input_data.__contains__('lrec'):
		input_data.lrec.remove()

	if input_data.__contains__('lsuppl'):
		input_data.lsuppl.remove()

	if db.root.__contains__('lmodel'):
		db.root.lmodel.remove()

	ldict.remove()

	print db
	db.close()


if __name__ == '__main__':
	#create_database()
	#init_dictionary()
	#init_dictionary(DATAPATH + 'processed/db.vocab')
	#init_dictionary(DATAPATH + 'processed/db-filtered.vocab') # Save the time initializing the filtered dictionary
	#filter_dictionary()
	#read_speaker_data()
	#read_document_data()
	#read_covariates_data()
	add_model_description()

	show_info()