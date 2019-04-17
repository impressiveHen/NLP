#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import collections
from math import log
import sys

# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)

class LangModel:
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in corpus:
            self.fit_sentence(s)
        self.norm()

    def perplexity(self, corpus):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        return pow(2.0, self.entropy(corpus))

    def entropy(self, corpus):
        num_words = 0.0
        sum_logprob = 0.0
        for s in corpus:
            num_words += len(s) + 1 # for EOS
            sum_logprob += self.logprob_sentence(s)
        return -(1.0/num_words)*(sum_logprob)

    def logprob_sentence(self, sentence):
        p = 0.0
        for i in xrange(len(sentence)):
            p += self.cond_logprob(sentence[i], sentence[:i])
        p += self.cond_logprob('END_OF_SENTENCE', sentence)
        return p

    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence): pass
    # optional, if there are any post-training steps (such as normalizing probabilities)
    def norm(self): pass
    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous): pass
    # required, the list of words the language model suports (including EOS)
    def vocab(self): pass
#==============================================================================

class Unigram(LangModel):
    def __init__(self, backoff = 0.000001):
        self.model = dict()
        self.lbackoff = log(backoff, 2)

    def inc_word(self, w):
        if w in self.model:
            self.model[w] += 1.0
        else:
            self.model[w] = 1.0

    def fit_sentence(self, sentence):
        for w in sentence:
            self.inc_word(w)
        self.inc_word('END_OF_SENTENCE')

    def norm(self):
        """Normalize and convert to log2-probs."""
        tot = 0.0
        for word in self.model:
            tot += self.model[word]
        ltot = log(tot, 2)
        for word in self.model:
            self.model[word] = log(self.model[word], 2) - ltot

    def cond_logprob(self, word, previous):
        if word in self.model:
            return self.model[word]
        else:
            return self.lbackoff

    def vocab(self):
        return self.model.keys()
#==============================================================================

class Trigram(LangModel):
    def __init__(self,threshold=4, alpha=0.1, normMeth="interpol"):
        assert isinstance(threshold,int)
        assert isinstance(alpha,float)
        assert isinstance(normMeth,str)
        assert normMeth in ["laplace","interpol"]
        self.model = dict()
        self.one = dict()
        self.two = dict()
        self.three = dict()
        self.threshold = threshold
        self.alpha = alpha
        self.normMeth = normMeth
        self.lmd = [0.0, 0.0, 1]

    def filter(self,corpus):
        for sentence in corpus:
            for word in sentence:
                self.one_inc_word(word)
        new_corpus = corpus
        unk_words = dict()
        for word in self.one:
            if self.one[word] < self.threshold:
                unk_words[word] = 0
        for i,sentence in enumerate(corpus):
            for j,word in enumerate(sentence):
                if(word in unk_words):
                    new_corpus[i][j] = 'UNK'
        self.one.clear()
        return new_corpus

    def get_normMeth(self):
        return self.normMeth

    def one_inc_word(self, word):
        if word in self.one:
            self.one[word] += 1.0
        else:
            self.one[word] = 1.0

    def bi_inc_word(self,word_1,word):
        if (word_1,word) in self.two:
            self.two[(word_1,word)] += 1.0
        else:
            self.two[(word_1,word)] = 1.0

    def tri_inc_word(self, word_2, word_1, word):
        if (word_2, word_1, word) in self.three:
            self.three[(word_2, word_1, word)] += 1.0
        else:
            self.three[(word_2, word_1, word)] = 1.0

    def fit_corpus(self, corpus):
        new_corpus = self.filter(corpus)
        for s in new_corpus:
            self.fit_sentence(s)
        self.norm()

    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence):
        add_sentence = ['*','*'] + sentence
        self.one_inc_word('*')
        for i, word in enumerate(sentence+['END_OF_SENTENCE']):
            self.tri_inc_word(add_sentence[i],add_sentence[i+1],word)
            self.bi_inc_word(add_sentence[i],add_sentence[i+1])
            self.one_inc_word(word)

    # optional, if there are any post-training steps (such as normalizing probabilities)
    # cal prob in self.model
    # laplace smoothing
    def norm(self):
        v = len(self.one)
        for tri in self.three:
            self.model[tri]=(self.three[tri]+1)/(self.two[tri[:2]]+self.alpha*v)

    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous):
        v = len(self.one)
        if len(previous)==0:
            previous = ['*','*'] + previous
        if len(previous)==1:
            previous = ['*'] + previous

        if self.normMeth == "laplace":
            if (previous[-2],previous[-1],word) in self.model:
                return log(self.model[(previous[-2],previous[-1],word)],2)
            else:
                return log(1/(self.alpha*v),2)
        else:
            return self.interpol_prob((previous[-2],previous[-1],word))

    def interpol_prob(self,tri):
        v=len(self.one)
        prob = 0
        lmd1 = self.lmd[0]
        lmd2 = self.lmd[1]
        lmd3 = self.lmd[2]
        if tri in self.model:
            prob = prob + lmd1*self.model[tri]
        if tri[:2] in self.two:
            prob = prob + lmd2*self.two[tri[:2]]/self.one[tri[0]]
        if tri[0] in self.one:
            prob = prob + lmd3*self.one[tri[0]]/v
        if prob==0:
            return log(0.000001,2)
        prob = log(prob,2)
        return prob

    def findLamdas(self,dev_corpus):
        min_per = float('inf')
        for lmd1 in [i*0.25 for i in range(5)]:
            for lmd2 in [i*0.25 for i in range(0,int((1-lmd1)*4)+1)]:
                lmd3 = 1 - lmd1 - lmd2
                temp_per = self.perplexity(dev_corpus)
                if temp_per < min_per:
                    min_per = temp_per
                    self.lmd[0] = lmd1
                    self.lmd[1] = lmd2
                    self.lmd[2] = lmd3



    # required, the list of words the language model suports (including EOS)
    def vocab(self):
        return self.one.keys()

if __name__ == '__main__':
    corpus = [['i', 'have', 'a', 'cat'],['i', 'have', 'love'],['i', 'have', 'a', 'bear']]
    trigram = Trigram()
    trigram.fit_corpus(corpus)
    print(trigram.one)
    print()
    print(trigram.two)
    print()
    print(trigram.three)
    print()
    print(trigram.model)
