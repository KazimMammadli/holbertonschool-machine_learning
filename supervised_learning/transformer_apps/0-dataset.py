#!/usr/bin/env python3
"""
Dataset module for machine translation (Portuguese to English).

This module defines the Dataset class which loads the TED Talks
Portuguese-English dataset and prepares pretrained tokenizers
for both languages.
"""

import tensorflow_datasets as tfds
from transformers import AutoTokenizer


class Dataset:
    """
    Dataset class for loading and preparing translation data.

    Attributes:
        data_train (tf.data.Dataset): Training split of dataset.
        data_valid (tf.data.Dataset): Validation split of dataset.
        tokenizer_pt (transformers tokenizer): Portuguese tokenizer.
        tokenizer_en (transformers tokenizer): English tokenizer.
    """

    def __init__(self):
        """
        Initializes the Dataset instance.

        Loads the TED HRLR Portuguese-to-English dataset and
        initializes pretrained tokenizers trained on the training set.
        """
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train',
            as_supervised=True
        )

        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True
        )

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates pretrained sub-word tokenizers for the dataset.

        Uses pretrained BERT tokenizers:
        - Portuguese: neuralmind/bert-base-portuguese-cased
        - English: bert-base-uncased

        The vocabulary size is limited to 2**13.

        Args:
            data (tf.data.Dataset): Dataset formatted as (pt, en).

        Returns:
            tokenizer_pt: Portuguese tokenizer.
            tokenizer_en: English tokenizer.
        """
        tokenizer_pt = AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased",
            model_max_length=2**13
        )

        tokenizer_en = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            model_max_length=2**13
        )

        return tokenizer_pt, tokenizer_en
