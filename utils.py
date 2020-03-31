#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This module contains auxiliary data structures.

import numpy as np
from datetime import datetime
from dateutil.parser import parse
from user_config import *


Token2Id = np.dtype([
		('token', 'S16'), # 16 chars appears to be enough
		('token_id', np.uint32),
		('token_freq', np.uint32)
		])

BoW = np.dtype([ # Bag of words
		('token_id', np.uint32),
		('token_freq', np.uint32)
		])


class TemporaryTable(object):
	"""A class for convinient manipulation of temporary data.
	Behaves just like an instance of the Table class."""
	_tables_num = 0

	def __init__(self, parent_slink, dtype, expectedrows=None):
		dbfile = parent_slink()._v_file
		self.dtype = dtype
		self.data = dbfile.createTable(parent_slink(), 'temp_%i' % TemporaryTable._tables_num, dtype,
					 'Temporary table %i' % TemporaryTable._tables_num, expectedrows=expectedrows)
		TemporaryTable._tables_num += 1 # Increase the counter


	def set(self, **kwargs):
		if self.dtype == Token2Id:
			tokenid = kwargs.get('tokenid')
			word_norm = kwargs.get('word_norm')
			frequency = kwargs.get('frequency')
			if not (tokenid or word_norm or frequency): # If arguments missing
				logger.error('Invalid keys! See the Token2Id type specification.')
				raise ValueError
			found = [row for row in self.data.where('token_id == tokenid')]
			if not found:
				# Extend the Token2ID table
				new_row = self.data.row
				new_row['token_id'] = tokenid
				new_row['token'] = word_norm
				new_row['token_freq'] = frequency
				new_row.append()
				self.data.flush()
			else:
				logger.error('Trying to assign a new token to an existing token_id')
				raise ValueError
		elif self.dtype == BoW:
			tokenid = kwargs.get('tokenid')
			frequency = kwargs.get('frequency')
			if not (tokenid or frequency): # If arguments missing
				logger.error('Invalid keys! See the BoW type specification.')
				raise ValueError
			# Assigns the new frequency value to tokenid
			found = False
			for row in self.data.where('token_id == tokenid'):
				row['token_freq'] = frequency
				row.update()
				found = True
			if not found:
				# Extend the LoW table
				new_row = self.data.row
				new_row['token_id'] = tokenid
				new_row['token_freq'] = frequency
				new_row.append()
			self.data.flush()
		else:
			raise NotImplementedError


	def __del__(self):
		self.data._f_remove() # ensure that the table is deleted properly
		TemporaryTable._tables_num -= 1


	@staticmethod
	def tables_num():
		return TemporaryTable._tables_num
#endclass TemporaryTable


def date2congress(current_date):
	"""Returns the Congress number for a given date."""
	date = parse(current_date)
	step = 2
	for year in xrange(FROMYEAR, TOYEAR, step):
		if parse('%d-01-03' % year) <= date <= parse('%d-01-03' % (year + step)):
			return (year - 1789) / 2 + 1


def date2sow(current_date, primary, primary_runoff, general, general_runoff):
	"""Return the current state of the world for a given date."""
	current_dt = parse(current_date)
	general_dt = parse(general)
	if not primary: # Sometimes primaries are not held
		if general_runoff:
			general_runoff_dt = parse(general_runoff)
			if general_runoff_dt <= current_dt:
				return 'general'
			elif (general_dt - current_dt).days <= GEN_BANDWIDTH:
				return 'general'
			else:
				return 'none'
		else:
			if (general_dt - current_dt).days <= GEN_BANDWIDTH:
				return 'general'
			else:
				return 'none'
	else:
		primary_dt = parse(primary)
		if general_runoff:
			general_runoff_dt = parse(general_runoff)
			if general_dt < current_dt <= general_runoff_dt:
				return 'general'
			elif primary_runoff:
				primary_runoff_dt = parse(primary_runoff)
				if (primary_runoff_dt < current_dt) and ((general_dt - current_dt).days <= GEN_BANDWIDTH):
					return 'general'
				elif primary_dt < current_dt <= primary_runoff_dt:
					return 'primary_runoff' # Or just 'primary'
				elif (primary_dt - current_dt).days <= PRIM_BANDWIDTH:
					return 'primary'
				else:
					return 'none'
			else:
				if (primary_dt < current_dt) and ((general_dt - current_dt).days <= GEN_BANDWIDTH):
					return 'general'
				elif (primary_dt - current_dt).days <= PRIM_BANDWIDTH:
					return 'primary'
				else:
					return 'none'
		else:
			if primary_runoff:
				primary_runoff_dt = parse(primary_runoff)
				if (primary_runoff_dt < current_dt) and ((general_dt - current_dt).days <= GEN_BANDWIDTH):
					return 'general'
				elif primary_dt < current_dt <= primary_runoff_dt:
					return 'primary_runoff' # Or just 'primary'
				elif (primary_dt - current_dt).days <= PRIM_BANDWIDTH:
					return 'primary'
				else:
					return 'none'
			else:
				if (primary_dt < current_dt) and ((general_dt - current_dt).days <= GEN_BANDWIDTH):
					return 'general'
				elif (primary_dt - current_dt).days <= PRIM_BANDWIDTH:
					return 'primary'
				else:
					return 'none'