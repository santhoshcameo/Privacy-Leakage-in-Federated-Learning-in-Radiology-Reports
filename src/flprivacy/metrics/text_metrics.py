"""Reconstruction-fidelity metrics for recovered clinical text.

Exact-sentence accuracy, sentence-level BLEU (S-BLEU) and ROUGE-L, all computed
on NFKC-normalised, lower-cased, whitespace-collapsed text so that scoring is
deterministic and tokenizer-agnostic.
"""
from __future__ import annotations
import re
import unicodedata
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

_SMOOTH = SmoothingFunction().method1


def normalize(text: str) -> str:
    """NFKC + lower-case + whitespace collapse (deterministic scoring form)."""
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def exact_match(reference: str, hypothesis: str) -> int:
    """1 if the normalised strings are identical, else 0."""
    return int(normalize(reference) == normalize(hypothesis))


def s_bleu(reference: str, hypothesis: str) -> float:
    """Sentence-level BLEU over 1-4 grams with method-1 smoothing."""
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    if not ref or not hyp:
        return 0.0
    return sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25),
                         smoothing_function=_SMOOTH)


def rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 from the longest common subsequence."""
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    m, n = len(ref), len(hyp)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if ref[i - 1] == hyp[j - 1] \
                else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    prec, rec = lcs / n, lcs / m
    return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
