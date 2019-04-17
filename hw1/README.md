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
        contains two language model implementations: Unigram, Trigram
class Unigram
An implementation of a simple *back-off* based unigram model is also included, and implements all of the functions
of the LangModel interface.

class Trigram
An implementation of trigram model with the option of using *Laplace Smoothing* or *Linear Interpolation*, and implements all           the functions of the LangModel interface
specify hyperparameters: threshold of filter, alpha of Laplace Smoothing, normMeth the Smoothing Method

generator.py: This file contains a simple word and sentence sampler for any language model. Since it supports arbitarily complex language models, it is not very efficient.
specify hyperparameter: temp to produce meaningful sentences

data.py: The primary file to run. This file contains methods to read the appropriate data files from the archive, train and evaluate all the unigram language model or trigram model (by calling "lm.py”), and generate sample sentences from all the models (by calling  “generator.py”). It also saves the result tables into LaTeX files.
