#!/usr/bin/env python3
'''creates a bag of words embedding matrix
'''


def bag_of_words(sentences, vocab=None):
    """Creates a bag of words embedding matrix"""
    cont_vector = CountVectorizer(vocabulary=vocab)
    embed = cont_vector.fit_transform(sentences)
    features = cont_vector.get_feature_names_out()
    return embed.toarray(), features
