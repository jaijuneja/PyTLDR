# -*- coding: utf-8 -*-
import unicodedata


def unicode_to_ascii(unicodestr):
    if isinstance(unicodestr, str):
        return unicodestr
    elif isinstance(unicodestr, unicode):
        return unicodedata.normalize('NFKD', unicodestr).encode('ascii', 'ignore')
    else:
        raise ValueError('Input text must be of type str or unicode.')


def parse_input(text, extractor='newspaper'):
    if isinstance(text, str) or isinstance(text, unicode):
        if text.startswith(('http://', 'https://')):
            # Input is a link - need to extract the text from html
            if extractor.lower() == 'goose':
                from goose import Goose
                urlparse = Goose()
                article = urlparse.extract(url=text)
                return unicode_to_ascii(article.cleaned_text)
            else:
                from newspaper import Article
                article = Article(text)
                article.download()
                article.parse()
                return unicode_to_ascii(article.text)
        elif text.endswith('.txt'):
            # Input is a file - need to read it
            textfile = open(text, 'rb')
            article = textfile.read()
            textfile.close()
            return unicode_to_ascii(article)
        else:
            # Input is a string containing the raw text
            return unicode_to_ascii(text)
    else:
        raise ValueError('Input text must be of type str or unicode.')