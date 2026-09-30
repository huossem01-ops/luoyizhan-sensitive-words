"""Mechanical normalization only; semantic variants must remain distinct."""

from __future__ import annotations

import re
import unicodedata

try:
    from opencc import OpenCC
except ImportError:  # pragma: no cover - optional for lightweight inspection
    OpenCC = None


_PUNCTUATION = re.compile(r"[\s\u3000，。！？、；：‘’“”\"'（）()【】\[\]《》<>·…—_\-]+")
_converter = OpenCC("t2s") if OpenCC else None


def to_simplified(value: str) -> str:
    return _converter.convert(value) if _converter else value


def normalize_term(term: str) -> str:
    value = unicodedata.normalize("NFKC", term).strip().lower()
    value = to_simplified(value)
    return _PUNCTUATION.sub("", value)


def char_ngrams(value: str, n: int = 2) -> set[str]:
    clean = normalize_term(value)
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[index:index + n] for index in range(len(clean) - n + 1)}
