#!/usr/bin/env python3
"""Cumulative n-gram BLEU: geometric mean of modified precisions and one BP."""

from collections import Counter

import numpy as np


def _ngram_counts(tokens, n):
    if len(tokens) < n:
        return Counter()
    return Counter(
        tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
    )


def _modified_precision(references, sentence, order):
    """Clipped modified n-gram precision for a single ``order``."""
    cand_len = len(sentence)
    denom = cand_len - order + 1
    if denom <= 0:
        return 0.0
    hyp = _ngram_counts(sentence, order)
    if not hyp:
        return 0.0
    clipped = sum(
        min(
            hyp[g],
            max((_ngram_counts(ref, order)[g] for ref in references), default=0),
        )
        for g in hyp
    )
    return clipped / denom


def cumulative_bleu(references, sentence, n):
    """
    Cumulative n-gram BLEU with evenly weighted gram orders via the geometric mean
    of modified precisions :math:`p_1 \\ldots p_n`, multiplied once by brevity penalty.

    Effective reference length is chosen as in ``uni_bleu``: minimal
    ``abs(len(reference) - len(sentence))``, tie-break shorter length.

    Args:
        references (list of list[str]): Reference tokenizations.
        sentence (list[str]): Hypothesis tokens.
        n (int): Largest n-gram order (must be >= 1).

    Returns:
        float: Score in ``[0, 1]``.
    """
    if n < 1 or not sentence:
        return float(0.0)

    cand_len = len(sentence)

    precisions = []
    for k in range(1, n + 1):
        precisions.append(_modified_precision(references, sentence, k))

    if not precisions or any(p <= 0 for p in precisions):
        geom = 0.0
    else:
        geom = float(np.exp(np.mean(np.log(precisions))))

    distances = [(abs(len(r) - cand_len), len(r)) for r in references]
    _, r = min(distances)

    if cand_len > r:
        bp = 1.0
    else:
        bp = float(np.exp(1 - r / cand_len))

    return bp * geom
