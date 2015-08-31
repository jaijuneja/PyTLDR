# -*- coding: utf-8 -*-
import ez_setup
ez_setup.use_setuptools(version='0.7')

from setuptools import setup
import os

PACKAGE_NAME = 'PyTLDR'
VERSION = '0.1.5'


def read(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    try:
        # Convert GitHub markdown to restructured text (needed for upload to PyPI)
        from pypandoc import convert
        return convert(filepath, 'rst')
    except ImportError:
        return open(filepath).read()

description = 'A module to perform automatic article summarization.'
try:
    long_description = read('README.md')
except IOError:
    long_description = description

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author='Jai Juneja',
    author_email='jai.juneja@gmail.com',
    description=description,
    license='BSD',
    keywords= [
        'summarizer', 'summarization', 'natural language processing', 'nlp',
        'machine learning', 'data mining', 'latent semantic analysis', 'lsa'
    ],
    url='https://github.com/jaijuneja/PyTLDR',
    packages=[
        'pytldr',
        'pytldr.nlp',
        'pytldr.summarize'
    ],
    long_description=long_description,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Filters',
        'Topic :: Text Processing :: Linguistic',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python'
    ],
    install_requires=[
        'numpy==1.8.0',
        'nltk==2.0.5',
        'scipy==0.13.2',
        'scikit-learn==0.15.2',
        'goose-extractor==1.0.25',
        'newspaper==0.0.9.8',
        'networkx==1.9.1'
    ],
    include_package_data=True,
    package_data={PACKAGE_NAME: ['stopwords/*.txt'],
                  '': ['README.md', 'ez_setup.py']},
    tests_require=[
        'nose',
        'coverage',
    ],
    test_suite='nose.collector'
)