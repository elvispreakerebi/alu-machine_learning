#!/usr/bin/env python3
"""Unigram (modified) BLEU with brevity penalty."""

from collections import Counter

import numpy as np


def uni_bleu(references, sentence):
    """
    Unigram BLEU: clipped unigram precision times brevity penalty.

    References:
        Papineni et al., BLEU: a Method for Automatic Evaluation of Machine Translation.

    Args:
        references (list of list[str]): Candidate reference tokenizations.
        sentence (list[str]): Hypothesis tokenization.

    Returns:
        float: Unigram BLEU score on ``[0, 1]``.
    """
    if not sentence:
        return float(0.0)

    cand_len = len(sentence)
    hyp_counts = Counter(sentence)

    clipped = sum(
        min(hyp_counts[w], max(ref.count(w) for ref in references))
        for w in hyp_counts
    )

    precision = clipped / cand_len

    distances = [(abs(len(r) - cand_len), len(r)) for r in references]
    _, r = min(distances)

    if cand_len > r:
        bp = 1.0
    else:
        bp = float(np.exp(1 - r / cand_len))

    return bp * precision
