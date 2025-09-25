"""Lightweight sequence augmentations for protein strings."""
from __future__ import annotations

import random
from typing import Optional, Tuple


def _ensure_rng(rng: Optional[random.Random] = None) -> random.Random:
    return rng if rng is not None else random.Random()


def random_span_mask(
    seq: str,
    p: float = 0.05,
    span: Tuple[int, int] = (3, 8),
    rng: Optional[random.Random] = None,
) -> str:
    """Mask random contiguous spans with 'X' tokens."""
    if not seq:
        return seq
    rng = _ensure_rng(rng)
    chars = list(seq)
    span_min, span_max = span
    i = 0
    length = len(chars)
    while i < length:
        if rng.random() < p:
            span_len = rng.randint(span_min, span_max)
            for j in range(i, min(length, i + span_len)):
                chars[j] = "X"
            i += span_len
        else:
            i += 1
    return "".join(chars)


def random_crop(
    seq: str,
    min_len: int = 80,
    rng: Optional[random.Random] = None,
) -> str:
    """Randomly crop the sequence while retaining at least ``min_len`` residues."""
    if len(seq) <= min_len:
        return seq
    rng = _ensure_rng(rng)
    target_len = rng.randint(min_len, len(seq))
    max_start = len(seq) - target_len
    start = rng.randint(0, max_start)
    return seq[start : start + target_len]


def jitter_tokens(
    seq: str,
    p: float = 0.01,
    rng: Optional[random.Random] = None,
) -> str:
    """Swap neighbouring tokens with probability ``p``."""
    if len(seq) < 2:
        return seq
    rng = _ensure_rng(rng)
    chars = list(seq)
    i = 0
    while i < len(chars) - 1:
        if rng.random() < p:
            j = i + 1 if rng.random() < 0.5 else max(0, i - 1)
            chars[i], chars[j] = chars[j], chars[i]
            i += 2
        else:
            i += 1
    return "".join(chars)


def make_two_views(seq: str, rng: Optional[random.Random] = None) -> Tuple[str, str]:
    """Produce two lightly augmented sequence variants."""
    rng = _ensure_rng(rng)
    base = seq.upper()
    if len(base) < 8:
        return base, base

    view_a = base
    view_b = base

    # View A: span mask + jitter
    view_a = random_span_mask(view_a, p=0.05, rng=rng)
    view_a = jitter_tokens(view_a, p=0.015, rng=rng)
    if len(view_a) > 120 and rng.random() < 0.3:
        view_a = random_crop(view_a, min_len=80, rng=rng)

    # View B: jitter first then optional span mask/crop
    view_b = jitter_tokens(view_b, p=0.02, rng=rng)
    if rng.random() < 0.5:
        view_b = random_span_mask(view_b, p=0.04, rng=rng)
    if len(view_b) > 120 and rng.random() < 0.3:
        view_b = random_crop(view_b, min_len=80, rng=rng)

    return view_a, view_b

