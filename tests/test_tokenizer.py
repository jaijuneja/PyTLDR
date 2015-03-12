# -*- coding: utf-8 -*-
import unittest
from pytldr.nlp import Tokenizer


class TestTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = Tokenizer('english')

    def test_tokenize_sentences(self):
        text = "This is a sentence. Lorem ipsum dolor sit amet...\n" \
               "Final sentence"
        expected_processed = [
            "sentenc",
            "lorem ipsum dolor sit amet",
            "final sentenc"
        ]
        expected_unprocessed = [
            "This is a sentence.",
            "Lorem ipsum dolor sit amet...",
            "Final sentence"
        ]

        processed_sentences, unprocessed_sentences = self.tokenizer.tokenize_sentences(text, word_threshold=0)

        self.assertEqual(expected_processed, processed_sentences)
        self.assertEqual(expected_unprocessed, unprocessed_sentences)

    def test_tokenize_words(self):
        text = "This is a sentence. Word."
        expected = ["This", "is", "a", "sentence", "Word"]
        result = self.tokenizer.tokenize_words(text)

        self.assertEqual(expected, result)

    def test_tokenize_paragraphs(self):
        text = """
            Here is a bunch of text.

            Another paragraph.
            Yet another paragraph.
            """
        expected = [
            "Here is a bunch of text.",
            "Another paragraph.",
            "Yet another paragraph."
        ]
        result = self.tokenizer.tokenize_paragraphs(text)
        self.assertEqual(expected, result)

    def test_stem(self):
        word = "stupidity"
        expected = "stupid"

        result = self.tokenizer.stem(word)
        self.assertEqual(expected, result)

    def test_stem_tokens(self):
        tokens = ["stupidity", "pieces", "and"]
        expected = ["stupid", "piec", "and"]

        result = self.tokenizer.stem_tokens(tokens)
        self.assertEqual(expected, result)

    def test_remove_stopwords(self):
        sentence = "Do you want to play a game"
        expected = "play game"
        result = self.tokenizer.remove_stopwords(sentence)
        self.assertEqual(expected, result)

        tokens = ["Do", "you", "want", "to", "play", "a", "game"]
        expected = ["play", "game"]
        result = self.tokenizer.remove_stopwords(tokens)
        self.assertEqual(expected, result)

    def test_load_stopwords(self):
        stop_words = ['the', 'and']
        tokenizer = Tokenizer(stopwords=stop_words)
        self.assertEqual(stop_words, tokenizer.stopwords)

    def test_language(self):
        self.assertRaises(ValueError, Tokenizer, "nonexistent language")

if __name__ == "__main__":
    unittest.main()