#!/usr/bin/env python
# -*- coding: utf-8 -*-


class DataContainer(object):
	"""A general interface. Uses soft links to tables to emulate the 'self' instance."""
	def __init__(self):
		raise NotImplementedError


	@staticmethod
	def length(slink):
		return slink().nrows


class TableWrapper(DataContainer):
	"""An interface for the Table wrappers."""
	cols = None


	@staticmethod
	def read(filestream=None):
		raise NotImplementedError


	@staticmethod
	def load_table(slink):
		raise NotImplementedError


	@staticmethod
	def update(slink, data):
		raise NotImplementedError


	@staticmethod
	def print_rows(slink):
		raise NotImplementedError
#endlcass TableWrapper