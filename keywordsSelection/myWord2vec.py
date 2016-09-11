import gensim, logging
import os
from sklearn.cluster import KMeans
import numpy as np

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

sentences = MySentences('./term/') # a memory-friendly iterator
model = gensim.models.word2vec(sentences , min_count=10, workers=4, size=200)
model.save('modelOut.txt')