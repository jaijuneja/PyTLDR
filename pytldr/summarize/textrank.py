# -*- coding: utf-8 -*-
from __future__ import division
import networkx
from .baseclass import BaseSummarizer


class TextRankSummarizer(BaseSummarizer):

    def summarize(self, text, length=5, weighting='frequency', norm=None):
        """
        """

        text = self._parse_input(text)

        sentences, unprocessed_sentences = self._tokenizer.tokenize_sentences(text)

        num_sentences = len(sentences)
        if 0 < length < 1:
            # length is a percentage - convert to number of sentences
            length = round(length * num_sentences)

        if length >= num_sentences:
            return unprocessed_sentences

        # Compute the word frequency matrix. If norm is set to 'l1' or 'l2' then words are normalized
        # by the length of their associated sentences (such that each vector of sentence terms sums to 1).
        word_matrix = self._compute_matrix(sentences, weighting=weighting, norm=norm)

        # Build the similarity graph by calculating the number of overlapping words between all
        # combinations of sentences.
        similarity_matrix = (word_matrix * word_matrix.T)

        similarity_graph = networkx.from_scipy_sparse_matrix(similarity_matrix)
        scores = networkx.pagerank(similarity_graph)

        ranked_sentences = sorted(
            ((score, ndx) for ndx, score in scores.items()), reverse=True
        )

        top_sentences = [ranked_sentences[i][1] for i in range(length)]
        top_sentences.sort()

        return [unprocessed_sentences[i] for i in top_sentences]