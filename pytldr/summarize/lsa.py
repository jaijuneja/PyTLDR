# -*- coding: utf-8 -*-
import numpy as np
from .baseclass import BaseSummarizer
from scipy.sparse.linalg import svds
from warnings import warn


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

    @classmethod
    def _validate_num_topics(cls, topics, sentences):
        # Determine the number of "linearly independent" sentences
        # This gives us an estimate for the rank of the matrix for which we will compute SVD
        sentences_set = set([frozenset(sentence.split(' ')) for sentence in sentences])
        est_matrix_rank = len(sentences_set)

        if est_matrix_rank <= 1:
            raise SvdRankException('The sentence matrix does not have sufficient rank to compute SVD')

        if topics > est_matrix_rank - 1:
            warn(
                'The parameter "topics" must be <= rank(sentence_matrix) - 1 to avoid rank '
                'deficiency in the SVD computation. The number of topics has been adjusted '
                'to equal rank(sentence_matrix) - 1 but this could result in a poor summary.',
                Warning
            )
            topics = est_matrix_rank - 1

        return topics


class SvdRankException(Exception):
    pass


class LsaSteinberger(BaseLsaSummarizer):

    def summarize(self, text, topics=4, length=5, binary_matrix=True, topic_sigma_threshold=0.5):
        """
        Implements the method of latent semantic analysis described by Steinberger and Jezek in the paper:

        J. Steinberger and K. Jezek (2004). Using latent semantic analysis in text summarization and summary evaluation.
        Proc. ISIM ’04, pp. 93–100.

        :param text: a string of text to be summarized, path to a text file, or URL starting with http
        :param topics: the number of topics/concepts covered in the input text (defines the degree of
        dimensionality reduction in the SVD step)
        :param length: the length of the output summary; either a number of sentences (e.g. 5) or a percentage
        of the original document (e.g. 0.5)
        :param binary_matrix: boolean value indicating whether the matrix of word counts should be binary
        (True by default)
        :param topic_sigma_threshold: filters out topics/concepts with a singular value less than this
        percentage of the largest singular value (must be between 0 and 1, 0.5 by default)
        :return: list of sentences for the summary
        """

        text = self._parse_input(text)

        sentences, unprocessed_sentences = self._tokenizer.tokenize_sentences(text)

        length = self._parse_summary_length(length, len(sentences))
        if length == len(sentences):
            return unprocessed_sentences

        topics = self._validate_num_topics(topics, sentences)

        # Generate a matrix of terms that appear in each sentence
        weighting = 'binary' if binary_matrix else 'frequency'
        sentence_matrix = self._compute_matrix(sentences, weighting=weighting)
        sentence_matrix = sentence_matrix.transpose()

        # Filter out negatives in the sparse matrix (need to do this on Vt for LSA method):
        sentence_matrix = sentence_matrix.multiply(sentence_matrix > 0)

        s, u, v = self._svd(sentence_matrix, num_concepts=topics)

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

    def summarize(self, text, topics=4, length=5, binary_matrix=True, topic_sigma_threshold=0):
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
        percentage of the largest singular value (must be between 0 and 1, 0 by default)
        :return: list of sentences for the summary
        """

        text = self._parse_input(text)

        sentences, unprocessed_sentences = self._tokenizer.tokenize_sentences(text)

        length = self._parse_summary_length(length, len(sentences))
        if length == len(sentences):
            return unprocessed_sentences

        topics = self._validate_num_topics(topics, sentences)

        weighting = 'binary' if binary_matrix else 'frequency'
        sentence_matrix = self._compute_matrix(sentences, weighting=weighting)
        sentence_matrix = sentence_matrix.transpose()

        # Filter out negatives in the sparse matrix (need to do this on Vt for LSA method):
        sentence_matrix = sentence_matrix.multiply(sentence_matrix > 0)

        s, u, v = self._svd(sentence_matrix, num_concepts=topics)

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