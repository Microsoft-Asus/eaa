#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This module contains implementation of the Dictionary class 
# needed for preprocessing of the raw Congressional Record data.
# 
# This code builds on gensim.corpora.dictionary.


import logging
import itertools

from gensim.corpora.dictionary import Dictionary

logger = logging.getLogger('eaa.corpora.cngrec_dictionary')


class CngrecDictionary(Dictionary):
    """Implementation of the Dictionary class."""
    def __init__(self, documents=None):
        super(CngrecDictionary, self).__init__(documents)


    @staticmethod
    def from_documents(documents):
        return CngrecDictionary(documents=documents)


    def add_documents(self, documents):
        """
        Build dictionary from a collection of documents. Each document is a list
        of tokens = **tokenized and normalized** utf-8 encoded strings.

        This is only a convenience wrapper for calling `doc2bow` on each document
        with `allow_update=True`.
        """
        for docno, document in enumerate(documents):
            logger.info("adding document #%i to %s" % (docno, self))
            _ = self.doc2bow(document, allow_update=True) # ignore the result, here we only care about updating token ids
        logger.info("built %s from %i documents (total %i corpus positions)" %
                     (self, self.num_docs, self.num_pos))


    def filter_extremes(self, no_below=0.05, no_above=0.95, keep_n=100000):
        """
        Filter out tokens that appear in

        1. less than `no_below` documents (fraction of total corpus size) or
        2. more than `no_above` documents (fraction of total corpus size).
        3. after (1) and (2), keep only the first `keep_n` most frequent tokens (or
           keep all if `None`).

        After the pruning, shrink resulting gaps in word ids.

        **Note**: Due to the gap shrinking, the same word may have a different
        word id before and after the call to this function!
        """
        no_below_abs = int(no_below * self.num_docs) # convert fractional threshold to absolute threshold
        no_above_abs = int(no_above * self.num_docs) 

        # determine which tokens to keep
        good_ids = (v for v in self.token2id.itervalues() if no_below_abs <= self.dfs[v] <= no_above_abs)
        good_ids = sorted(good_ids, key=self.dfs.get, reverse=True)
        if keep_n is not None:
            good_ids = good_ids[:keep_n]
        logger.info("keeping %i tokens which were in no less than %i and no more than %i (=%.1f%%) documents" %
                     (len(good_ids), no_below_abs, no_above_abs, 100.0 * no_above))

        # do the actual filtering, then rebuild dictionary to remove gaps in ids
        self.filter_tokens(good_ids=good_ids)
        self.compactify()
        logger.info("resulting dictionary: %s" % self)
#endclass CngrecDictionary

