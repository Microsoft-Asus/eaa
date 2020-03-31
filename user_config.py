# -*- coding: utf-8 -*-

# Input data parameters

CORPORA = './' # change this
DATAPATH = './data/'
CHAMBERS = ('House', 'Senate')
CONGRESSES = (108, 109, 110, 111)
FROMYEAR = 2003
TOYEAR = 2011
PRIM_BANDWIDTH = 60 # In days
GEN_BANDWIDTH = 180

# Chamber and party codes
CHAMBER_S = 'S' # Senate
CHAMBER_H = 'H' # House
PARTY_R = 'R' # Republican
PARTY_D = 'D' # Democratic

# Vocabulary constraints

NO_BELOW = 0.005
NO_ABOVE = 0.95
VOCAB_SIZE = 4000
NUM_TOPICS = 44
NUM_COVARIATES = 6 # Including a const
NUM_DOCS = 15418
WPT = 18 # Number of most representative words per each topic
DPT = 5 # Number of random documents per each topic for manual check

# VB constants
EPS = 1e-9 # E step tolerance
EPS_0 = 1e-6 # ELBO tolerance
TOL = 1e-6 # Optimization tolerance
MAX_ITER = 10 # The maximum number of iterations in the function f minimization
PASSES = 20
ITERS = 3 # The maximum number of iterations in the E step

# The DW-NOMINATE data representation format

DW_FIELDNAMES = ('Congress', 'ICPSR ID', 'State Code', 'District', 'Party', 'Name', '1st Dimension', '2nd Dimension', '1st Dimension S.E.', '2nd Dimension S.E.', 'Correlation Log-likelihood', 'Votes Classification Errors', 'Geometric Mean Probability')
STATES = {
	'AL': '41',
	'AK': '81',
	'AZ': '61',
	'AR': '42',
	'CA': '71',
	'CO': '62',
	'CT': '01',
	'DE': '11',
	'FL': '43',
	'GA': '44',
	'HI': '82',
	'ID': '63',
	'IL': '21',
	'IN': '22',
	'IA': '31',
	'KS': '32',
	'KY': '51',
	'LA': '45',
	'ME': '02',
	'MD': '52',
	'MA': '03',
	'MI': '23',
	'MN': '33',
	'MS': '46',
	'MO': '34',
	'MT': '64',
	'NE': '35',
	'NV': '65',
	'NH': '04',
	'NJ': '12',
	'NM': '66',
	'NY': '13',
	'NC': '47',
	'ND': '36',
	'OH': '24',
	'OK': '53',
	'OR': '72',
	'PA': '14',
	'RI': '05',
	'SC': '48',
	'SD': '37',
	'TN': '54',
	'TX': '49',
	'UT': '67',
	'VA': '40',
	'VT': '06',
	'WA': '73',
	'WV': '56',
	'WI': '25',
	'WY': '68',
}
PARTIES = {
	'Democrat': '100',
	'Republican': '200',
	'Independent': '328'
}

# Output data parameters

DELIM = '\t'
DATEFORMAT = '%Y-%m-%d %H:%M'
PROGRESS_CNT = 5e3