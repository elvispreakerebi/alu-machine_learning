#!/usr/bin/env python3
"""Bag-of-words sentence embeddings."""

import re

import numpy as np


def _tokenize(sentence):
    """Lowercase sentence, extract alphabetic runs, drop single-letter tokens."""
    words = re.findall(r"[a-z]+", sentence.lower())
    return [w for w in words if len(w) > 1]


def bag_of_words(sentences, vocab=None):
    """
    Build a bag-of-words count matrix.

    Words are contiguous runs of lowercase letters; non-letters separate tokens.
    Single-letter tokens (e.g. ``s`` from ``children's``) are ignored.

    Args:
        sentences (list): Strings to encode.
        vocab (list | None): Feature names in sorted order desired; if ``None``,
            use every token from ``sentences`` (after tokenization rules above),
            sorted alphabetically.

    Returns:
        tuple:
            embeddings (np.ndarray): Shape ``(len(sentences), len(features))``.
            features (list): Vocabulary ordering used as columns (strings).
    """
    if vocab is None:
        seen = []
        for s in sentences:
            for w in _tokenize(s):
                seen.append(w)
        features = sorted(set(seen))
    else:
        features = list(vocab)

    word_to_col = {w: i for i, w in enumerate(features)}
    embeddings = np.zeros((len(sentences), len(features)), dtype=np.int64)

    for si, sentence in enumerate(sentences):
        counts = {}
        for w in _tokenize(sentence):
            if w in word_to_col:
                counts[w] = counts.get(w, 0) + 1
        for w, n in counts.items():
            embeddings[si, word_to_col[w]] = n

    return embeddings, features
