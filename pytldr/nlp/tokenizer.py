# -*- coding: utf-8 -*-
import os.path
from nltk.stem import SnowballStemmer
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from string import punctuation


class Tokenizer(object):

    def __init__(self, language='english'):
        self.stopwords = self._load_stopwords(language)
        self.stemmer = SnowballStemmer(language)

    @staticmethod
    def _load_stopwords(language):
        # Load stop-words
        stopwords_dir = 'stopwords/{}.txt'.format(language.lower())
        application_root = os.path.dirname(__file__)
        stopwords_dir = os.path.join(application_root, stopwords_dir)

        try:
            with open(stopwords_dir, 'rb') as stopwords_file:
                stopwords = [word.strip('\n') for word in stopwords_file.readlines()]
        except IOError:
            stopwords = []

        return stopwords

    def remove_stopwords(self, tokens):
        """Remove all stopwords from a list of word tokens or a string of text."""
        if isinstance(tokens, (list, tuple)):
            return [word for word in tokens if word not in self.stopwords]
        else:
            return ' '.join(
                [word for word in tokens.split(' ') if word not in self.stopwords]
            )

    def stem(self, word):
        return self.stemmer.stem(word)

    def stem_tokens(self, tokens):
        """Perform snowball (Porter2) stemming on a list of word tokens."""
        return [self.stem(word) for word in tokens]

    @staticmethod
    def strip_punctuation(text, exclude='', include=''):
        """Strip leading and trailing punctuation from an input string."""
        chars_to_strip = ''.join(
            set(list(punctuation)).union(set(list(include))) - set(list(exclude))
        )
        return text.strip(chars_to_strip)

    def tokenize_words(self, text):
        """Tokenize an input string into a list of words (with punctuation removed)."""
        return [
            self.strip_punctuation(word) for word in text.split(' ')
            if self.strip_punctuation(word)
        ]

    def sanitize_text(self, text):
        tokens = self.tokenize_words(text.lower())
        tokens = self.remove_stopwords(tokens)
        tokens = self.stem_tokens(tokens)
        sanitized_text = ' '.join(tokens)
        return sanitized_text

    @staticmethod
    def _remove_whitespace(text):
        while True:
            old_text = text
            text = text.replace('  ', ' ')
            if text == old_text:
                return text

    def tokenize_sentences(self, text, word_threshold=5):
        punkt_params = PunktParameters()
        punkt_params.abbrev_types = {
            'dr', 'vs', 'mr', 'mrs', 'ms', 'prof', 'mt', 'inc', 'i.e', 'e.g'
        }
        sentence_splitter = PunktSentenceTokenizer(punkt_params)

        # 1. TOKENIZE "UNPROCESSED" SENTENCES FOR DISPLAY
        # Need to adjust quotations for correct sentence splitting
        text_unprocessed = text.replace('?"', '? "').replace('!"', '! "').replace('."', '. "')
        # Treat line breaks as end of sentence (needed in cases where titles don't have a full stop)
        text_unprocessed = text_unprocessed.replace('\n', ' . ')
        # Perform sentence splitting
        unprocessed_sentences = sentence_splitter.tokenize(text_unprocessed)
        # Now that sentences have been split we can return them back to normal
        unprocessed_sentences = [sen.replace('? " ', '?" ').replace('! " ', '!" ').replace('. " ', '." ')
                                 for sen in unprocessed_sentences]
        # Remove excess whitespace so that sentences display correctly
        unprocessed_sentences = [self._remove_whitespace(sen) for sen in unprocessed_sentences]

        # 2. PROCESS THE SENTENCES TO PERFORM STEMMING, STOPWORDS REMOVAL ETC. FOR MATRIX COMPUTATION
        processed_sentences = [self.sanitize_text(sen) for sen in unprocessed_sentences]

        # Sentences should contain at least 'word_threshold' significant terms
        filter_sentences = [i for i in range(len(processed_sentences))
                            if len(processed_sentences[i].replace('.', '').split(' ')) > word_threshold]

        processed_sentences = [processed_sentences[i] for i in filter_sentences]
        unprocessed_sentences = [unprocessed_sentences[i] for i in filter_sentences]

        return processed_sentences, unprocessed_sentences