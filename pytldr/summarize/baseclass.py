# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from ..nlp.preprocess import parse_input
from ..nlp.tokenizer import Tokenizer


class BaseSummarizer(object):

    def __init__(self, tokenizer=Tokenizer('english')):
        self._tokenizer = tokenizer

    def summarize(self, text, num_sentences=5):
        raise NotImplementedError('This method needs to be implemented in a child class')

    @classmethod
    def _compute_matrix(cls, sentences, weighting='frequency', norm=None):
        """
        Compute the matrix of term frequencies given a list of sentences
        """

        if norm not in ('l1', 'l2', None):
            raise ValueError('Parameter "norm" can only take values "l1", "l2" or None')

        # Initialise vectorizer to convert text documents into matrix of token counts
        if weighting.lower() == 'binary':
            vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 1), binary=True, stop_words=None)
        elif weighting.lower() == 'frequency':
            vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 1), binary=False, stop_words=None)
        elif weighting.lower() == 'tfidf':
            vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 1), stop_words=None)
        else:
            raise ValueError('Parameter "method" must take one of the values "binary", "frequency" or "tfidf".')

        # Extract word features from sentences using sparse vectorizer
        frequency_matrix = vectorizer.fit_transform(sentences).astype(float)

        # Normalize the term vectors (i.e. each row adds to 1)
        if norm in ('l1', 'l2'):
            frequency_matrix = normalize(frequency_matrix, norm=norm, axis=1)
        elif norm is not None:
            raise ValueError('Parameter "norm" can only take values "l1", "l2" or None')

        return frequency_matrix

    @classmethod
    def _parse_input(cls, text):
        return parse_input(text)

    @classmethod
    def _parse_summary_length(cls, length, num_sentences):
        if length < 0 or not isinstance(length, (int, float)):
            raise ValueError('Parameter "length" must be a positive number')
        elif 0 < length < 1:
            # length is a percentage - convert to number of sentences
            return int(round(length * num_sentences))
        elif length >= num_sentences:
            return num_sentences
        else:
            return int(length)