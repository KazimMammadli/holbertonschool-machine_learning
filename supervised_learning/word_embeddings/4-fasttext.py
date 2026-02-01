#!/usr/bin/env python3
'''creates and trains a genism fastText model
'''
from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    '''creates and trains a genism fastText model'''
    model = FastText(sentences=sentences, size=size, window=window,
                     min_count=min_count, workers=workers, seed=seed,
                     sg=cbow, iter=iterations,
                     negative=negative)
    model.train(sentences=sentences, total_examples=model.corpus_count,
                epochs=model.epochs)
    return model
