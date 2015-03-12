import unittest
import warnings
from pytldr.summarize.baseclass import BaseSummarizer
from pytldr.summarize import LsaOzsoy, LsaSteinberger, RelevanceSummarizer, TextRankSummarizer


class TestSummarizer(unittest.TestCase):
    """
    Generic test class for all summarizers
    """
    __test__ = False
    summarizer = None
    text =  """
            Lorem ipsum dolor sit amet, consectetur adipiscing elit.
            Ut interdum sed purus quis vehicula.
            Aliquam nec congue mi, a commodo elit.
            Praesent porta lacus velit, at consequat quam vestibulum in.
            Nunc rutrum sapien volutpat augue porttitor vulputate ac sit amet metus.
            """
    expected_summary = [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Ut interdum sed purus quis vehicula.",
        "Aliquam nec congue mi, a commodo elit.",
        "Praesent porta lacus velit, at consequat quam vestibulum in.",
        "Nunc rutrum sapien volutpat augue porttitor vulputate ac sit amet metus."
    ]

    def test_length(self):
        # When length is 5 we just expect the input text (which is 5 sentences)
        summary = self.summarizer.summarize(self.text, length=5)
        self.assertEqual(len(summary), 5)

        # When length is >5 the output should simply be suppressed to 5 sentences
        summary = self.summarizer.summarize(self.text, length=10)
        self.assertEqual(len(summary), 5)

        # When length is between 0 and 1 we use it as a percentage
        summary = self.summarizer.summarize(self.text, length=0.6)
        self.assertEqual(len(summary), 3)

    def test_summarize(self):
        summary = self.summarizer.summarize(self.text, length=5)
        self.assertEqual(summary, self.expected_summary)

    def test_matrix_shape(self):
        sentences = ["bunch long words", "more long words", "hello dude"]
        unique_terms = 6
        num_sentences = 3
        matrix = self.summarizer._compute_matrix(sentences)
        self.assertEqual(matrix.shape, (num_sentences, unique_terms))

    def assertWarns(self, warning, callable, *args, **kwds):
        """Catch any warnings"""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter('always')

            result = callable(*args, **kwds)

            self.assertTrue(any(item.category == warning for item in warning_list))


class TestLsaOzsoySummarizer(TestSummarizer):
    __test__ = True

    def setUp(self):
        self.summarizer = LsaOzsoy()

    def test_rank_deficiency(self):
        # If too many topics are specified for SVD computation a warning should be raised
        topics = 100
        # Length needs to be less than number of topics AND less than the original text for warning to appear
        length = 4
        self.assertWarns(Warning, self.summarizer.summarize,
                         self.text, topics=topics, length=length)


class TestLsaSteinbergerSummarizer(TestSummarizer):
    __test__ = True

    def setUp(self):
        self.summarizer = LsaSteinberger()

    def test_rank_deficiency(self):
        # If too many topics are specified for SVD computation a warning should be raised
        topics = 100
        # Length needs to be less than number of topics AND less than the original text for warning to appear
        length = 4
        self.assertWarns(Warning, self.summarizer.summarize,
                         self.text, topics=topics, length=length)


class TestRelevanceSummarizer(TestSummarizer):
    __test__ = True

    def setUp(self):
        self.summarizer = RelevanceSummarizer()


class TestTextRankSummarizer(TestSummarizer):
    __test__ = True

    def setUp(self):
        self.summarizer = TextRankSummarizer()


class TestBaseSummarizer(unittest.TestCase):

    def test_summarize_abstract(self):
        self.assertRaises(TypeError, BaseSummarizer, '')