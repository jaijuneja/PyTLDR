# PyTLDR: Automatic Text Summarization in Python

[![Build Status](https://travis-ci.org/jaijuneja/PyTLDR.svg?branch=master)](https://travis-ci.org/jaijuneja/PyTLDR) [![PyPI version](https://badge.fury.io/py/pytldr.svg)](https://pypi.python.org/pypi/pytldr)

A Python module to perform automatic summarization of articles, text files and web pages.

## License

Copyright 2014 Jai Juneja.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).

## Installation

### Using pip or easy_install

You can download the latest release version using `pip` or `easy_install`:

```
pip install pytldr
```

### Latest development version
You can alternatively download the latest development version directly from GitHub:

```
git clone https://github.com/jaijuneja/PyTLDR.git
```

Change into the root directory:

```
cd pytldr
```

Then install the package:

```
python setup.py install
```

## Usage

A simple sample program using the PyTLDR module can be found at `https://github.com/jaijuneja/PyTLDR/blob/master/example.py`

In its current form, this module contains three distinct implementations of automatic text summarization:

* Using the TextRank algorithm (based on PageRank)
* Using Latent Semantic Analysis
* Using a sentence relevance score
 
Note that all three of the above implementations are extractive - that is, they simply extract and display the most relevant sentences from the input text. They do not formulate their own sentences (such algorithms are known as "abstractive", and are still at a primitive stage).

### Sentence tokenization

PyTLDR comes with a built-in sentence tokenizer that is used for summarization. The tokenizer performs stemming in several languages as well as stop-word removal. You may also specify your own list of stop-words.

```python
from pytldr.nlp import Tokenizer

tokenizer = Tokenizer(language='english', stopwords=None, stemming=True)
# Note that if stopwords=None then the tokenizer loads stopwords from a bundled data-set
# You can alternatively specify a text file or provide a list of words
```

Note that the tokenizer is the only input required to initialize a summarizer object, as shown below.

### TextRank Summarization

Ranks sentences using the PageRank algorithm, where "votes" or "in-links" are represented by words shared between sentences.

```python
from pytldr.summarize import TextRankSummarizer
from pytldr.nlp import Tokenizer

tokenizer = Tokenizer('english')
summarizer = TextRankSummarizer(tokenizer)

# If you don't specify a tokenizer when intiializing a summarizer then the
# English tokenizer will be used by default
summarizer = TextRankSummarizer()  # English tokenizer used

# This object creates a summary using the summarize method:
# e.g. summarizer.summarize(text, length=5, weighting='frequency', norm=None)

# The length parameter specifies the length of the summary, either as a
# number of sentences, or a percentage of the original text

# The summarizer can take as input...
# 1. A string:
summary = summarizer.summarize("Some long article bla bla...", length=4)

# 2. A text file:
summary = summarizer.summarize("/path/to/file.txt", length=0.25)
# Above summary is a quarter of the length of the original text

# 3. A URL (must start with http://):
summary = summarizer.summarize("http://newsite.com/some_article")
```

### Latent Semantic Analysis (LSA) Summarization

Reduces the dimensionality of the article into several "topic" clusters using singular value decomposition, and selects the sentences that are most relevant to these topics. This is a rather more abstract summarization algorithm.

This module comes packaged with two distinct implementations of the LSA algorithm, as described in two academic papers:

* J. Steinberger and K. Jezek (2004). Using latent semantic analysis in text summarization and summary evaluation.
* Ozsoy, M., Alpaslan, F., and Cicekli, I. (2011). Text summarization using latent semantic analysis.

The more recent Ozsoy et al. implentation is called by default, but both classes have the same interface.

```python
from pytldr.summarize import LsaSummarizer, LsaOzsoy, LsaSteinberger

summarizer = LsaOzsoy()
summarizer = LsaSteinberger()
summarizer = LsaSummarizer()  # This is identical to the LsaOzsoy object

summary = summarizer.summarize(
    text, topics=4, length=5, binary_matrix=True, topic_sigma_threshold=0.5
)

# topics specifies the number of topics to cluster the article into.
# topic_sigma_threshold removes all topics with a singular value less than a given
# percentage of the largest singular value.
```

### Relevance Score Summarization

This method computes and ranks the cosine similarity between each sentence vector and the overall document, removing the most relevant sentence at each iteration. It closely follows the approach described in the paper:

* Y. Gong and X. Liu (2001). Generic text summarization using relevance measure and latent semantic analysis.

```python
from pytldr.summarize import RelevanceSummarizer

summarizer = RelevanceSummarizer()
summary = summarizer.summarize(text, length=5, binary_matrix=True):
```

### More help

You can read the documentation for each of the above implementations by typing the following into your python console:

```python
help(TextRankSummarizer)
help(LsaSummarizer)
help(RelevanceSummarizer)
```

## Contact

If you have any questions or have encountered an error, feel free to contact me at `jai -dot- juneja -at- gmail -dot- com`.