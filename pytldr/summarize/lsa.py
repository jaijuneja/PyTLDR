# -*- coding: utf-8 -*-
import numpy as np
from .baseclass import BaseSummarizer
from scipy.sparse.linalg import svds


class BaseLsaSummarizer(BaseSummarizer):
    """
    This is an abstract base class for summarizers using the LSA method.
    """

    @classmethod
    def _svd(cls, matrix, num_concepts=5):
        """
        Perform singular value decomposition for dimensionality reduction of the input matrix.
        """
        u, s, v = svds(matrix, k=num_concepts)
        return u, s, v


class LsaSteinberger(BaseLsaSummarizer):

    def summarize(self, text, topics=5, length=5, binary_matrix=True, topic_sigma_threshold=0.5):
        """
        Implements the method of latent semantic analysis described by Steinberger and Jezek in the paper:

        J. Steinberger and K. Jezek. Using latent semantic analysis in text summarization and summary evaluation.
        Proc. ISIM ’04, 2004, pp. 93–100.

        :param text: a string of text to be summarized, path to a text file, or URL starting with http
        :param topics: the number of topics/concepts covered in the input text (defines the degree of
        dimensionality reduction in the SVD step)
        :param length: the length of the output summary; either a number of sentences (5) or a percentage
        of the original document (e.g. 0.5)
        :param binary_matrix: boolean value indicating whether the matrix of word counts should be binary
        (True by default)
        :param topic_sigma_threshold: filters out topics/concepts with a singular value less than this
        percentage of the largest singular value (must be between 0 and 1)
        :return: list of sentences for the summary
        """

        text = self._parse_input(text)

        sentences, unprocessed_sentences = self._tokenizer.tokenize_sentences(text)

        if 0 < length < 1:
            # length is a percentage - convert to number of sentences
            length = round(length * len(sentences))

        if length >= len(sentences):
            return unprocessed_sentences

        weighting = 'binary' if binary_matrix else 'frequency'
        matrix = self._compute_matrix(sentences, weighting=weighting)

        # Filter out negatives in the sparse matrix (need to do this on Vt for LSA method):
        matrix = matrix.multiply(matrix > 0)

        s, u, v = self._svd(matrix, num_concepts=topics)

        # Only consider topics/concepts whose singular values are half of the largest singular value
        if 1 <= topic_sigma_threshold < 0:
            raise ValueError('Parameter topic_sigma_threshold must take a value between 0 and 1')

        sigma_threshold = max(u) * topic_sigma_threshold
        u[u < sigma_threshold] = 0  # Set all other singular values to zero

        # Build a "length vector" containing the length (i.e. saliency) of each sentence
        saliency_vec = np.dot(np.square(u), np.square(v))

        top_sentences = saliency_vec.argsort()[-length:][::-1]
        # Return the sentences in the order in which they appear in the document
        top_sentences.sort()

        return [unprocessed_sentences[i] for i in top_sentences]


class LsaOzsoy(BaseLsaSummarizer):

    def summarize(self, text, topics=5, length=5, binary_matrix=True, topic_sigma_threshold=0):
        """
        Implements the "cross method" of latent semantic analysis described by Ozsoy et al. in the paper:

        Ozsoy, M., Alpaslan, F., and Cicekli, I. (2011). Text summarization using latent semantic analysis.
        Journal of Information Science, 37(4), 405-417.

        :param text: a string of text to be summarized, path to a text file, or URL starting with http
        :param topics: the number of topics/concepts covered in the input text (defines the degree of
        dimensionality reduction in the SVD step)
        :param length: the length of the output summary; either a number of sentences (5) or a percentage
        of the original document (e.g. 0.5)
        :param binary_matrix: boolean value indicating whether the matrix of word counts should be binary
        (True by default)
        :param topic_sigma_threshold: filters out topics/concepts with a singular value less than this
        percentage of the largest singular value (must be between 0 and 1)
        :return: list of sentences for the summary
        """

        text = self._parse_input(text)

        sentences, unprocessed_sentences = self._tokenizer.tokenize_sentences(text)

        if 0 < length < 1:
            # length is a percentage - convert to number of sentences
            length = round(length * len(sentences))

        if length >= len(sentences):
            return unprocessed_sentences

        weighting = 'binary' if binary_matrix else 'frequency'
        matrix = self._compute_matrix(sentences, weighting=weighting)

        # Filter out negatives in the sparse matrix (need to do this on Vt for LSA method):
        matrix = matrix.multiply(matrix > 0)

        s, u, v = self._svd(matrix, num_concepts=topics)

        # Get the average sentence score for each topic (i.e. each row in matrix v)
        topic_averages = v.mean(axis=1)

        # Set sentences whose scores fall below the topic average to zero
        # This removes less related sentences from each concept
        for topic_ndx, topic_avg in enumerate(topic_averages):
            v[topic_ndx, v[topic_ndx, :] <= topic_avg] = 0

        # Only consider topics/concepts whose singular values are a specified % of largest singular value
        if 1 <= topic_sigma_threshold < 0:
            raise ValueError('Parameter topic_sigma_threshold must take a value between 0 and 1')

        sigma_threshold = max(u) * topic_sigma_threshold
        u[u < sigma_threshold] = 0  # Set all other singular values to zero

        # Build a "length vector" containing the length (i.e. saliency) of each sentence
        saliency_vec = np.dot(np.square(u), np.square(v))

        top_sentences = saliency_vec.argsort()[-length:][::-1]
        # Return the sentences in the order in which they appear in the document
        top_sentences.sort()

        return [unprocessed_sentences[i] for i in top_sentences]


# Default LsaSummarizer just uses the Ozsoy method
LsaSummarizer = LsaOzsoy