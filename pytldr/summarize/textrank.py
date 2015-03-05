# -*- coding: utf-8 -*-
import numpy as np
from .baseclass import BaseSummarizer


class TextRankSummarizer(BaseSummarizer):

    def summarize(self, text, length=5):
        """
        """

        text = self._parse_input(text)

        sentences, unprocessed_sentences = self._tokenizer.tokenize_sentences(text)

        if 0 < length < 1:
            # length is a percentage - convert to number of sentences
            length = round(length * len(sentences))

        if length >= len(sentences):
            return unprocessed_sentences

        matrix = self._compute_matrix(sentences, weighting='frequency')
        matrix = matrix.transpose()
