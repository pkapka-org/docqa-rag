"""Check every sentence of an answer against the passages it cites.

Models sometimes cite the wrong passage or forget to cite at all. For each sentence:
  * citations to passages that do not exist are dropped;
  * a citation is dropped when another passage clearly supports the sentence better;
  * a sentence with no citation gets one if a passage clearly supports it.
"Supports" means the passage shares the sentence's words and contains all of its numbers.
"""

from __future__ import annotations

import re

_MARK = re.compile(r"\s*\[\s*(\d+(?:\s*[,;]\s*\d+)*)\s*\]")
_AFTER_STOP = re.compile(r"([.!?])((?:\s*\[[\d,;\s]+\])+)")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(*])")
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:[.,:]\d+)*")
_STOP = set(
    "a an and are as at be been but by can do does for from has have if in into is it its may must "
    "no not of on or only per should so than that the their them then there they this to up was "
    "were what when which who will with within you your yes".split()
)

SUPPORT = 0.5


def _tokens(text: str) -> tuple[set[str], set[str]]:
    """Content words (crudely stemmed) and numbers."""
    words, numbers = set(), set()
    for token in _WORD.findall(text.lower()):
        if token[0].isdigit():
            numbers.add(token.replace(",", ""))
        elif token not in _STOP:
            words.add(re.sub(r"(?:ing|ed|es|s)$", "", token) if len(token) > 4 else token)
    return words, numbers


def support(sentence: str, passage: str) -> float:
    """0..1. Zero when the sentence has a number the passage does not contain."""
    words, numbers = _tokens(sentence)
    passage_words, passage_numbers = _tokens(passage)
    if numbers - passage_numbers:
        return 0.0
    total = len(words) + len(numbers)
    return (len(words & passage_words) + len(numbers)) / total if total else 0.0


def ground(text: str, passages: list[str]) -> tuple[str, list[int], bool]:
    """Return (text with corrected citations, cited numbers in order, anything corrected?)."""
    cited: list[int] = []
    changed = False
    text = _AFTER_STOP.sub(r"\2\1", text)  # "Fact. [1]" -> "Fact [1]."

    lines = []
    for line in text.splitlines():
        sentences = []
        for sentence in _SENTENCE_END.split(line):
            claimed = [int(n) for m in _MARK.findall(sentence) for n in re.findall(r"\d+", m)]
            bare = _MARK.sub("", sentence).rstrip()
            numbers = _pick(bare, list(dict.fromkeys(claimed)), passages)
            changed |= numbers != list(dict.fromkeys(claimed))
            cited += [n for n in numbers if n not in cited]
            sentences.append(_attach(bare, numbers))
        lines.append(" ".join(sentences))
    return "\n".join(lines).strip(), cited, changed


def _pick(sentence: str, claimed: list[int], passages: list[str]) -> list[int]:
    if not sentence or not _tokens(sentence)[0] | _tokens(sentence)[1]:
        return []
    scores = [support(sentence, p) for p in passages]
    best = max(scores, default=0.0)
    valid = [n for n in claimed if 1 <= n <= len(passages)]
    if best < SUPPORT:
        return valid  # a paraphrase or a fact built from several passages: trust the model
    kept = [n for n in valid if scores[n - 1] >= best * 0.6]
    return kept or [scores.index(best) + 1]


def _attach(sentence: str, numbers: list[int]) -> str:
    if not numbers:
        return sentence
    marks = "".join(f"[{n}]" for n in numbers)
    body = sentence.rstrip(".!?")
    return f"{body} {marks}{sentence[len(body):]}"
