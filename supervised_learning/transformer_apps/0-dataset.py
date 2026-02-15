#!/usr/bin/env python3
"""
Dataset module for Portuguese to English machine translation.

Defines the Dataset class which loads the TED HRLR translation
dataset and initializes pretrained tokenizers.
"""

import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Loads and prepares the translation dataset.

    Attributes:
        data_train: Training dataset split.
        data_valid: Validation dataset split.
        tokenizer_pt: Portuguese tokenizer.
        tokenizer_en: English tokenizer.
    """

    def __init__(self):
        """
        Initializes the Dataset instance.

        Loads the TED HRLR Portuguese-English dataset and creates
        pretrained tokenizers.
        """
        self.data_train = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split="train",
            as_supervised=True
        )

        self.data_valid = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split="validation",
            as_supervised=True
        )

        self.tokenizer_pt, self.tokenizer_en = \
            self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates pretrained sub-word tokenizers.

        Uses:
        - neuralmind/bert-base-portuguese-cased
        - bert-base-uncased

        Args:
            data: tf.data.Dataset formatted as (pt, en).

        Returns:
            tokenizer_pt: Portuguese tokenizer.
            tokenizer_en: English tokenizer.
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased",
            model_max_length=2 ** 13
        )

        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            model_max_length=2 ** 13
        )

        return tokenizer_pt, tokenizer_en
