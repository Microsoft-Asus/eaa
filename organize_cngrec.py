#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# This script organizes the Congressional Record by sessions, 
# speakers, and datetime, representing the data as plain text documents.

import csv
import codecs, dateutil.parser
import os
import datetime
from lxml import etree
from user_config import *


def convert_time(unix_time):
	return datetime.datetime.fromtimestamp(int(unix_time)).strftime('%Y-%m-%d-%H:%M')


def organize_cngrec():
	root = etree.parse(DATAPATH + '108-111_congresses.xml')
	errors = open('./log/errors.log', 'w')
	for cngr in CONGRESSES:
		session = str(cngr)
		if not os.path.isdir(DATAPATH + session):
			os.mkdir(DATAPATH + session)
		people = root.xpath('//congress-history/people[@session=' + session + ']/person')
		for person in people:
			govtrack_id = person.attrib['id']
			icpsr_id = person.attrib['icpsrid']
			speeches_path = CORPORA + session + '/index.cr.person/' + govtrack_id + '.xml'
			if os.path.isfile(speeches_path):
				speeches = etree.parse(speeches_path).xpath('//speeches/cr')
				if not os.path.isdir(DATAPATH + session + '/' + icpsr_id):
					os.mkdir(DATAPATH + session + '/' + icpsr_id)
				for speech in speeches:
					record = speech.attrib['file'].split('/')[1]
					body = speech.attrib['where']
					date = speech.attrib['when']
					#date = dateutil.parser.parse(speech.attrib['datetime'])
					#date = date.strftime(DATEFORMAT)
					if date:
						date = convert_time(speech.attrib['when'])
						text_path = CORPORA + session + '/cr/' + record + '.xml'
						if os.path.isfile(text_path):
							text = etree.parse(text_path).xpath('//record/speaking[@speaker=' + govtrack_id + ']/paragraph')
							output_path = DATAPATH + session + '/' + icpsr_id + '/' + session + '_' + icpsr_id + '_' + date
							output = open(output_path, 'a+')
							for speaking in text:
								if speaking.text: output.write(speaking.text + '\n')
							output.close()
						else:
							errors.write('Path not found: %s\n' % text_path)
			else: 
				errors.write('Path not found: %s\n' % speeches_path)
	errors.close()


if __name__ == '__main__':
	organize_cngrec()