# -*- coding: utf-8 -*-
import numpy as np
from .baseclass import BaseSummarizer
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import CountVectorizer


class BaseLsaSummarizer(BaseSummarizer):
    """
    This is an abstract base class for summarizers using the LSA method.
    """
    @staticmethod
    def _compute_matrix(sentences, binary=True):
        """
        Compute the matrix of term frequencies given a list of sentences
        """
        # Initialise vectorizer to convert text documents into matrix of token counts
        vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 1), binary=binary)

        # Extract word features from sentences using sparse vectorizer
        frequency_matrix = vectorizer.fit_transform(sentences).transpose()
        return frequency_matrix.astype(float)

    @classmethod
    def _svd(cls, matrix, num_concepts=5):
        """
        Perform singular value decomposition for dimensionality reduction of the input matrix.
        """
        u, s, v = svds(matrix, k=num_concepts)
        return u, s, v


class LsaSteinberger(BaseLsaSummarizer):

    def summarize(self, text, num_sentences=5):

        text = self.parse_input(text)

        sentences, unprocessed_sentences = self._tokenizer.tokenize_sentences(text)

        matrix = self._compute_matrix(sentences)

        # Filter out negatives in the sparse matrix (need to do this on Vt for LSA method):
        matrix = matrix.multiply(matrix > 0)

        s, u, v = self._svd(matrix)

        # Only consider topics/concepts whose singular values are half of the largest singular value
        sigma_threshold = max(u) / 2
        u[u < sigma_threshold] = 0  # Set all other singular values to zero

        # Build a "length vector" containing the length (i.e. saliency) of each sentence
        length_vec = np.dot(np.square(u), np.square(v))

        top_sentences = length_vec.argsort()[-num_sentences:][::-1]
        top_sentences.sort()

        return [unprocessed_sentences[i] for i in top_sentences]


class LsaOzsoy(BaseLsaSummarizer):
    pass