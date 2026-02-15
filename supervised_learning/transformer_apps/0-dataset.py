#!/usr/bin/env python3
"""Dataset class for machine translation (Portuguese to English)"""
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """Loads and prepares a dataset for machine translation"""

    def __init__(self):
        """
        Creates instance attributes:
        - data_train: ted_hrlr_translate/pt_to_en train split
        - data_valid: ted_hrlr_translate/pt_to_en validate split
        - tokenizer_pt: Portuguese tokenizer trained on training set
        - tokenizer_en: English tokenizer trained on training set
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
        Creates sub-word tokenizers for the dataset using pre-trained models.

        Args:
            data: tf.data.Dataset whose examples are formatted as (pt, en)
                  pt: tf.Tensor containing the Portuguese sentence
                  en: tf.Tensor containing the corresponding English sentence

        Returns:
            tokenizer_pt: Portuguese tokenizer (BertTokenizerFast)
            tokenizer_en: English tokenizer (BertTokenizerFast)
        """
        max_vocab_size = 2 ** 13  # 8192

        # Collect raw sentences from the dataset
        pt_sentences = []
        en_sentences = []
        for pt, en in data:
            pt_sentences.append(pt.numpy().decode('utf-8'))
            en_sentences.append(en.numpy().decode('utf-8'))

        # Load pre-trained tokenizers
        pt_tokenizer = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        en_tokenizer = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased'
        )

        # Train Portuguese tokenizer on the collected sentences
        tokenizer_pt = pt_tokenizer.train_new_from_iterator(
            pt_sentences,
            vocab_size=max_vocab_size
        )

        # Train English tokenizer on the collected sentences
        tokenizer_en = en_tokenizer.train_new_from_iterator(
            en_sentences,
            vocab_size=max_vocab_size
        )

        return tokenizer_pt, tokenizer_en
