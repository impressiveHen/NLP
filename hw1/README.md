# Comparing Unigram and Trigram model for sentence generation

## Installation:
```bash
python version 2.0 above
pip install numpy
pip install tabulate
pip install sklearn
```

## Usage:
``` python
python data.py
```
        
## File:
lm.py: This file describes the higher level interface for a language model, and contains functions to train, query, and evaluate it. 
        
        class Unigram
        An implementation of a simple *back-off* based unigram model is also included, that implements all of the functions
        of the interface.
        
        class Trigram
        An implementation of trigram model with the option of using *Laplace Smoothing* or *Linear Interpolation*, which implements all           the functions of the LangModel interface

generator.py: This file contains a simple word and sentence sampler for any language model. Since it supports arbitarily complex language models, it is not very efficient. If this sampler is incredibly slow for your language model, you can consider implementing your own (by caching the conditional probability tables, for example, instead of computing it for every word).

data.py: The primary file to run. This file contains methods to read the appropriate data files from the archive, train and evaluate all the unigram language models (by calling “lm.py”), and generate sample sentences from all the models (by calling  “generator.py”). It also saves the result tables into LaTeX files.
