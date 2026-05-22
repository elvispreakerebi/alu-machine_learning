#!/usr/bin/env python3
"""TF–IDF sentence embeddings (smooth IDF + L2-normalized rows)."""

from collections import Counter

import numpy as np

_tokenize = __import__("0-bag_of_words")._tokenize


def tf_idf(sentences, vocab=None):
    """
    Build TF–IDF vectors for each sentence and L2-normalize each row.

    Term frequency ``tf(term, doc)`` is ``count(term) / len(tokens)`` where
    ``tokens`` follows the same preprocessing as ``bag_of_words``: lowercase
    letter runs with single-letter runs removed.

    Inverse document frequency (smooth, natural logarithm): ``df(term)`` is
    how many sentences contain ``term``. If ``df`` is ``0`` (never in the corpus),
    the weight is ``0``. Else ``idf = 1 + log((N + 1) / (df + 1))``, with
    ``N = len(sentences)``.

    Args:
        sentences (list of str): Corpus.
        vocab (list[str] | None): Feature order; when ``None``, use every token
            that appears after preprocessing, sorted alphabetically.

    Returns:
        tuple:
            embeddings (np.ndarray): Shape ``(len(sentences), len(features))``.
            features (list[str]): Column labels.
    """
    sentences = list(sentences)
    n_docs = len(sentences)

    if vocab is None:
        corpus_tokens = []
        for s in sentences:
            corpus_tokens.extend(_tokenize(s))
        features = sorted(set(corpus_tokens))
    else:
        features = list(vocab)

    n_feats = len(features)
    embeddings = np.zeros((n_docs, n_feats), dtype=float)

    token_rows = [_tokenize(sent) for sent in sentences]

    dfs = np.array([
        sum(1 for toks in token_rows if word in toks)
        for word in features], dtype=int)

    for i, sentence in enumerate(sentences):
        tokens = _tokenize(sentence)
        if not tokens:
            continue
        denom = len(tokens)
        counts = Counter(tokens)
        for j, word in enumerate(features):
            df = dfs[j]
            if df == 0:
                continue
            cnt = counts.get(word, 0)
            if cnt == 0:
                continue
            idf = 1 + np.log((n_docs + 1) / (df + 1))
            embeddings[i, j] = (cnt / denom) * idf

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.divide(
        embeddings,
        norms,
        out=np.zeros_like(embeddings),
        where=norms > 0,
    )
    return embeddings, features
