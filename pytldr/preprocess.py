# encoding=utf-8
from sklearn.feature_extraction.text import CountVectorizer
import nltk.tokenize
from nltk.stem import SnowballStemmer
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from string import punctuation
import os.path
import unicodedata


def unicode_to_ascii(unicodestr):
    if isinstance(unicodestr, str):
        return unicodestr
    elif isinstance(unicodestr, unicode):
        return unicodedata.normalize('NFKD', unicodestr).encode('ascii', 'ignore')
    else:
        raise Exception('Input text is not of type unicode or str.')


def parse_input(text):
    if isinstance(text, str) or isinstance(text, unicode):
        if text.startswith('http'):
            return
        elif text.endswith(('.doc', '.docx', '.txt')):
            return
        else:
            return text


def load_stopwords(language='english'):
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


def remove_stopwords(tokens, language='english'):
    """Remove all stopwords from a list of word tokens."""
    stopwords = load_stopwords(language)
    return [word for word in tokens if word not in stopwords]


def remove_punctuation(text, replacement=' ', exclude=""):
    """Remove punctuation from an input string."""
    text = text.replace("'", "")  # Single quote always stripped out
    for p in set(list(punctuation)) - set(list(exclude + '.')):  # Full stops never stripped out
        text = text.replace(p, replacement)

    text = ' '.join(text.split())  # Remove excess whitespace
    return text


def stem_tokens(tokens, language='english'):
    """Perform snowball (Porter2) stemming on a list of word tokens."""
    stemmer = SnowballStemmer(language)
    return [stemmer.stem(word) for word in tokens]


def sanitize_text(text, language='english'):
    text = remove_punctuation(text)
    tokens = tokenize_words(text.lower())
    tokens = remove_stopwords(tokens, language)
    tokens = stem_tokens(tokens, language)
    sanitized_text = ' '.join(tokens)
    return unicode_to_ascii(sanitized_text)


def remove_whitespace(text):
    while True:
        old_text = text
        text = text.replace('  ', ' ')
        if text == old_text:
            return text


def compute_frequency_matrix(sentences, binary=True):
    # Initialise vectorizer to convert text documents into matrix of token counts
    vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 1), binary=binary)

    # Extract word features from sentences using sparse vectorizer
    frequency_matrix = vectorizer.fit_transform(sentences).transpose()
    return frequency_matrix.astype(float)


def tokenize_words(text):
    """Tokenize an input string into a list of words (with punctuation removed)."""
    text = text.lower()
    punctuation_removed = remove_punctuation(text)
    tokens = nltk.word_tokenize(punctuation_removed)
    return tokens


def tokenize_sentences(text, word_threshold=6, language='english'):
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
    unprocessed_sentences = [remove_whitespace(sen) for sen in unprocessed_sentences]

    # 2. PROCESS THE SENTENCES TO PERFORM STEMMING, STOPWORDS REMOVAL ETC. FOR MATRIX COMPUTATION
    processed_sentences = [sanitize_text(s, language=language) for s in unprocessed_sentences]

    # Sentences should contain at least 'word_threshold' significant terms
    filter_sentences = [i for i in range(len(processed_sentences))
                        if len(processed_sentences[i].replace('.', '').split(' ')) > word_threshold]

    processed_sentences = [processed_sentences[i] for i in filter_sentences]
    unprocessed_sentences = [unprocessed_sentences[i] for i in filter_sentences]

    return processed_sentences, unprocessed_sentences