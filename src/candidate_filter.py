"""Deterministic filters for words and fixed/common phrases."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
URL = re.compile(r"(?:https?://|www\.|\.(?:com|cn|net|org)(?:/|$))", re.I)
HTML = re.compile(r"<[^>]+>|&(?:nbsp|amp|lt|gt);", re.I)
RANDOM_DIGITS = re.compile(r"\d{5,}")
REPEATED_CHAR = re.compile(r"^(.)\1{1,}$")
SENTENCE_PUNCT = re.compile(r"[。！？!?；;]")
CONTROL = re.compile(r"[\x00-\x1f\x7f]")
CONTEXT_PREFIX = re.compile(r"^(?:昨天|今天|明天|周[一二三四五六日天]|早上|上午|中午|下午|晚上|深夜)")
PRONOUN_START = re.compile(r"^(?:我|你|他|她|我们|你们|他们|她们)")
SENTENCE_END = re.compile(r"(?:了吗|了吗|吗|呢|吧|啊|呀|嘛|么)$")
SHORT_CLAUSE = re.compile(
    r"^(?:什么是|为什么|怎么会|是否|有没有|而我|而你|跟我|跟你|和我|和你|"
    r"那时候|那晚|听后|死后|花园里|你的|我的|他的|她的)"
)


def is_person_name(value: str) -> bool:
    """Conservatively reject terms Jieba recognizes as one complete person name."""
    if not (2 <= len(value) <= 4) or len(CJK.findall(value)) != len(value):
        return False
    import jieba.posseg as pseg

    tokens = list(pseg.cut(value))
    return len(tokens) == 1 and tokens[0].word == value and tokens[0].flag.startswith("nr")


@dataclass(frozen=True)
class FilterResult:
    keep: bool
    term: str
    length: int
    reason: str


def clean_surface(term: str) -> str:
    value = unicodedata.normalize("NFKC", term).strip()
    return re.sub(r"\s+", "", value)


def evaluate_candidate(term: str, config: dict) -> FilterResult:
    value = clean_surface(term)
    length = len(value)
    allow = set(config.get("allow_terms", []))
    if value in allow:
        return FilterResult(True, value, length, "allowlist")
    if not value:
        return FilterResult(False, value, length, "empty")
    if CONTROL.search(value) or URL.search(value) or HTML.search(value):
        return FilterResult(False, value, length, "url_html_control")
    if RANDOM_DIGITS.search(value) or value.isdigit():
        return FilterResult(False, value, length, "numeric_noise")
    if REPEATED_CHAR.fullmatch(value):
        return FilterResult(False, value, length, "repeated_character_noise")
    if SENTENCE_PUNCT.search(value):
        return FilterResult(False, value, length, "sentence_punctuation")
    if length < int(config["min_length"]):
        return FilterResult(False, value, length, "too_short")
    if length > int(config["hard_max_length"]):
        return FilterResult(False, value, length, "over_30_hard_limit")
    if length > int(config["soft_max_length"]):
        return FilterResult(False, value, length, "over_16_default_limit")
    cjk_count = len(CJK.findall(value))
    if cjk_count / max(length, 1) < float(config["min_cjk_ratio"]):
        return FilterResult(False, value, length, "low_cjk_ratio")
    if config.get("reject_person_names", False) and is_person_name(value):
        return FilterResult(False, value, length, "person_name")
    if SHORT_CLAUSE.search(value):
        return FilterResult(False, value, length, "sentence_fragment")
    if length >= 7 and (CONTEXT_PREFIX.search(value) or PRONOUN_START.search(value)):
        return FilterResult(False, value, length, "sentence_like_context")
    if length >= 7 and SENTENCE_END.search(value):
        return FilterResult(False, value, length, "sentence_like_ending")
    if value.count("，") + value.count(",") + value.count("、") > 1:
        return FilterResult(False, value, length, "too_many_punctuation_marks")
    return FilterResult(True, value, length, "accepted")
