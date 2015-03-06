# -*- coding: utf-8 -*-
import numpy as np
from .baseclass import BaseSummarizer


class RelevanceSummarizer(BaseSummarizer):

    def summarize(self, text, length=5, binary_matrix=True):
        """
        Implements the method of summarization by relevance score, as described by Gong and Liu in the paper:

        Y. Gong and X. Liu (2001). Generic text summarization using relevance measure and latent semantic analysis.
        Proceedings of the 24th International Conference on Research in Information Retrieval (SIGIR ’01),
        pp. 19–25.

        This method computes and ranks the cosine similarity between each sentence vector and the overall
        document.

        :param text: a string of text to be summarized, path to a text file, or URL starting with http
        :param length: the length of the output summary; either a number of sentences (e.g. 5) or a percentage
        of the original document (e.g. 0.5)
        :param binary_matrix: boolean value indicating whether the matrix of word counts should be binary
        (True by default)
        :return: list of sentences for the summary
        """

        text = self._parse_input(text)

        sentences, unprocessed_sentences = self._tokenizer.tokenize_sentences(text)

        length = self._parse_summary_length(length, len(sentences))
        if length == len(sentences):
            return unprocessed_sentences

        matrix = self._compute_matrix(sentences, weighting='frequency')

        # Sum occurrences of terms over all sentences to obtain document frequency
        doc_frequency = matrix.sum(axis=0)

        if binary_matrix:
            matrix = (matrix != 0).astype(int)

        summary_sentences = []
        for _ in range(length):
            # Take the inner product of each sentence vector with the document vector
            sentence_scores = matrix.dot(doc_frequency.transpose())
            sentence_scores = np.array(sentence_scores.T)[0]

            # Grab the top sentence and add it to the summary
            top_sentence = sentence_scores.argsort()[-1]
            summary_sentences.append(top_sentence)

            # Remove all terms that appear in the top sentence from the document
            terms_in_top_sentence = (matrix[top_sentence, :] != 0).toarray()
            doc_frequency[terms_in_top_sentence] = 0

            # Remove the top sentence from consideration by setting all its elements to zero
            # This does the same as matrix[top_sentence, :] = 0, but is much faster for sparse matrices
            matrix.data[matrix.indptr[top_sentence]:matrix.indptr[top_sentence+1]] = 0
            matrix.eliminate_zeros()

        # Return the sentences in the order in which they appear in the document
        summary_sentences.sort()
        return [unprocessed_sentences[i] for i in summary_sentences]