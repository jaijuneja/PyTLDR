# -*- coding: utf-8 -*-
from __future__ import division
import networkx
from .baseclass import BaseSummarizer


class TextRankSummarizer(BaseSummarizer):

    def summarize(self, text, length=5, weighting='frequency', norm=None):
        """
        Implements the TextRank summarization algorithm, which follows closely to the PageRank algorithm for ranking
        web pages.

        :param text: a string of text to be summarized, path to a text file, or URL starting with http
        :param length: the length of the output summary; either a number of sentences (e.g. 5) or a percentage
        of the original document (e.g. 0.5)
        :param weighting: 'frequency', 'binary' or 'tfidf' weighting of sentence terms ('frequency' by default)
        :param norm: if 'l1' or 'l2', normalizes words by the length of their associated sentence to "down-weight"
        the voting power of long sentences (None by default)
        :return: list of sentences for the summary
        """

        text = self._parse_input(text)

        sentences, unprocessed_sentences = self._tokenizer.tokenize_sentences(text)

        length = self._parse_summary_length(length, len(sentences))
        if length == len(sentences):
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