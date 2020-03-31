#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This script merges the DW-NOMINATE and GovTrack data.
#
# Missing attributes:
# ICPSRID District 1st Dimension
#
# KEY = (Last name, State) allows to uniquely identify legislators

import csv
import codecs
from collections import OrderedDict
from lxml import etree
from user_config import *


def augment_people_xml_bulk():
	matching = open(DATAPATH + 'matching', 'w')
	for cngr in CONGRESSES:
		congress = str(cngr)
		people = etree.parse(CORPORA + congress + '/people.xml')
		for person in people.xpath('//people/person'):
			# Delete 
			#if not (person.attrib['state'] in STATES.keys()):
			#	person.getparent().remove(person)
			# Convert to the DW-NOMINATE format
			lastname = person.attrib['lastname'].upper()
			state_code = str(int(STATES[person.attrib['state']]))
			party = PARTIES[person.getchildren()[0].attrib['party']]
			district = ''
			if 'district' in person.attrib:
				district = person.attrib['district']
			dwnominate = csv.DictReader(
				open(DATAPATH + 'dw-congress-108-111.csv', 'r'),
				skipinitialspace=True, 
				restval='', dialect='excel')
			if district != '':
				search = [
					(item['ICPSR ID'], item['State Code'], item['District'], item['Party'], item['Name'], item['1st Dimension']) \
					for item in dwnominate \
					if (item['Congress'] == congress) & (item['State Code'] == state_code) \
					& (item['Party'] == party) & (item['District'] == district) \
					& (item['Name'] == lastname)
				]
				if len(search) == 0:
					matching.write('%s %s %s %s %s has not been matched\n' % (congress, state_code, district, party, lastname))
				elif len(search) > 1:
					matching.write('%s %s %s %s %s has more than one occurence\n' % (congress, state_code, district, party, lastname))
				else:
					person.attrib['icpsrid'] = search[0][0]
					person.attrib['dw1stdim'] = search[0][5]
			else:
				search = [
					(item['ICPSR ID'], item['State Code'], item['District'], item['Party'], item['Name'], item['1st Dimension']) \
					for item in dwnominate if (item['Congress'] == congress) \
					& (item['State Code'] == state_code) & (item['Party'] == party) \
					& (item['Name'] == lastname)
				]
				if len(search) == 0:
					matching.write('%s %s %s %s has not been matched\n' % (congress, state_code, party, lastname))
				elif len(search) > 1:
					matching.write('%s %s %s %s has more than one occurence\n' % (congress, state_code, party, lastname))
				else:
					person.attrib['icpsrid'] = search[0][0]
					person.attrib['district'] = search[0][2]
					person.attrib['dw1stdim'] = search[0][5]
		output = codecs.open(CORPORA + congress + '/people-augmented.xml', 'wb')
		output.write(etree.tostring(people, pretty_print=True))
		output.close()
	matching.close()


def augment_people_xml():
	matching = open(DATAPATH + 'matching', 'w')
	people = etree.parse(DATAPATH + '108-111_congresses.xml')
	for cngr in CONGRESSES:
		congress = str(cngr)
		for person in people.xpath('//congress-history/people[@session=' + congress + ']/person'):
			# Delete 
			#if not (person.attrib['state'] in STATES.keys()):
			#	person.getparent().remove(person)
			# Convert to the DW-NOMINATE format
			lastname = person.attrib['lastname'].upper()
			state_code = str(int(STATES[person.attrib['state']]))
			party = PARTIES[person.getchildren()[0].attrib['party']]
			district = ''
			if 'district' in person.attrib:
				district = person.attrib['district']
			dwnominate = csv.DictReader(
				open(DATAPATH + 'dw-congress-108-111.csv', 'r'),
				skipinitialspace=True, 
				restval='', dialect='excel')
			if district != '':
				search = [
					(item['ICPSR ID'], item['State Code'], item['District'], item['Party'], item['Name'], item['1st Dimension']) \
					for item in dwnominate \
					if (item['Congress'] == congress) & (item['State Code'] == state_code) \
					& (item['Party'] == party) & (item['District'] == district) \
					& (item['Name'] == lastname)
				]
				if len(search) == 0:
					matching.write('%s %s %s %s %s has not been matched\n' % (congress, state_code, district, party, lastname))
				elif len(search) > 1:
					matching.write('%s %s %s %s %s has more than one occurence\n' % (congress, state_code, district, party, lastname))
				else:
					person.attrib['icpsrid'] = search[0][0]
					person.attrib['dw1stdim'] = search[0][5]
			else:
				search = [
					(item['ICPSR ID'], item['State Code'], item['District'], item['Party'], item['Name'], item['1st Dimension']) \
					for item in dwnominate if (item['Congress'] == congress) \
					& (item['State Code'] == state_code) & (item['Party'] == party) \
					& (item['Name'] == lastname)
				]
				if len(search) == 0:
					matching.write('%s %s %s %s has not been matched\n' % (congress, state_code, party, lastname))
				elif len(search) > 1:
					matching.write('%s %s %s %s has more than one occurence\n' % (congress, state_code, party, lastname))
				else:
					person.attrib['icpsrid'] = search[0][0]
					person.attrib['district'] = search[0][2]
					person.attrib['dw1stdim'] = search[0][5]
		output = codecs.open(DATAPATH + 'people-augmented.xml', 'wb')
		output.write(etree.tostring(people, pretty_print=True))
		output.close()
	matching.close()


if __name__ == '__main__':
	augment_people_xml()