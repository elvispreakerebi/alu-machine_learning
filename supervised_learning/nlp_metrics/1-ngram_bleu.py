#!/usr/bin/env python3
"""N-gram (modified) BLEU with brevity penalty."""

from collections import Counter

import numpy as np


def _ngram_counts(tokens, n):
    """Return ``Counter`` of length-``n`` tuples from ``tokens``."""
    if len(tokens) < n:
        return Counter()
    return Counter(
        tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
    )


def ngram_bleu(references, sentence, n):
    """
    N-gram BLEU: modified n-gram precision times the same brevity penalty as
    ``uni_bleu`` (effective reference length by minimal distance to hypothesis
    length; tie-break smaller reference length).

    Args:
        references (list of list[str]): Reference tokenizations.
        sentence (list[str]): Hypothesis tokenization.
        n (int): N-gram order (``1`` recovers unigram-style precision over words).

    Returns:
        float: N-gram BLEU score on ``[0, 1]``.
    """
    if n < 1 or not sentence:
        return float(0.0)

    cand_len = len(sentence)

    denom = cand_len - n + 1
    if denom <= 0:
        return float(0.0)

    hyp_counts = _ngram_counts(sentence, n)

    clipped = sum(
        min(
            hyp_counts[gram],
            max((_ngram_counts(ref, n)[gram] for ref in references), default=0),
        )
        for gram in hyp_counts
    )

    precision = clipped / denom

    distances = [(abs(len(r) - cand_len), len(r)) for r in references]
    _, r = min(distances)

    if cand_len > r:
        bp = 1.0
    else:
        bp = float(np.exp(1 - r / cand_len))

    return bp * precision
