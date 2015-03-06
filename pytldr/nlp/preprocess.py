# -*- coding: utf-8 -*-
import unicodedata


def unicode_to_ascii(unicodestr):
    if isinstance(unicodestr, str):
        return unicodestr
    elif isinstance(unicodestr, unicode):
        return unicodedata.normalize('NFKD', unicodestr).encode('ascii', 'ignore')
    else:
        raise ValueError('Input text must be of type str or unicode.')