"""Spelling Correction Logic: Unigram/Bigram Modeling, Non-Word and Real-Word Error Correction."""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from .candidate_gen import (
    generate_candidates_edit1,
    generate_candidates_sym_del,
)


class BigramLM:
    """Smoothed Bigram Language Model for contextual scoring and real-word error correction."""

    def __init__(self, bigram_counts: Counter, unigram_counts: Counter, vocab_size: int, k: float = 0.05):
        self.bigrams = bigram_counts
        self.unigrams = unigram_counts
        self.vocab_size = max(vocab_size, 1)
        self.k = k

    def log_prob(self, prev_word: str, curr_word: str) -> float:
        """Returns smoothed log probability log P(curr_word | prev_word)."""
        w1 = prev_word.lower()
        w2 = curr_word.lower()

        c_bi = self.bigrams.get((w1, w2), 0)
        c_uni = self.unigrams.get(w1, 0)

        prob = (c_bi + self.k) / (c_uni + self.k * self.vocab_size)
        return math.log(prob)

    def score_phrase(self, words: List[str]) -> float:
        """Returns total log probability of a sequence of words with boundary markers."""
        if not words:
            return 0.0
        padded = ["<s>"] + [w.lower() for w in words] + ["</s>"]
        total_log_prob = 0.0
        for i in range(1, len(padded)):
            total_log_prob += self.log_prob(padded[i - 1], padded[i])
        return total_log_prob


def build_vocabulary_and_unigram(
    corpus: List[List[str]],
) -> Tuple[Set[str], Dict[str, float], Counter]:
    """Creates a vocabulary and frequency distribution (unigram model).

    Args:
        corpus: List of tokenized sentences.

    Returns:
        Tuple of (vocabulary_set, unigram_probabilities, unigram_counts).
    """
    counts: Counter = Counter()
    total_tokens = 0

    for sentence in corpus:
        for word in sentence:
            cleaned = word.strip().lower()
            if cleaned:
                counts[cleaned] += 1
                total_tokens += 1

    vocab = set(counts.keys())
    vocab_size = max(len(vocab), 1)

    # Smoothed unigram probability P(w)
    unigram_probs: Dict[str, float] = {}
    for word, count in counts.items():
        unigram_probs[word] = (count + 1) / (total_tokens + vocab_size)

    return vocab, unigram_probs, counts


def build_bigram_model(
    corpus: List[List[str]],
    k: float = 0.05,
) -> BigramLM:
    """Creates a bigram probability model for handling context-aware real-word errors.

    Args:
        corpus: List of tokenized sentences.
        k: Add-k smoothing parameter.

    Returns:
        Trained BigramLM instance.
    """
    unigrams: Counter = Counter()
    bigrams: Counter = Counter()
    vocab: Set[str] = set()

    for sentence in corpus:
        cleaned = [w.strip().lower() for w in sentence if w.strip()]
        if not cleaned:
            continue

        padded = ["<s>"] + cleaned + ["</s>"]
        for i in range(len(padded)):
            w = padded[i]
            unigrams[w] += 1
            vocab.add(w)
            if i > 0:
                bigrams[(padded[i - 1], w)] += 1

    return BigramLM(bigrams, unigrams, len(vocab), k=k)


def correct_non_word(
    word: str,
    vocab: Set[str],
    unigram_model: Dict[str, float],
    sym_del_dict: Dict[str, List[str]],
    method: str = "B",
) -> str:
    """Corrects a non-word error using unigram probabilities to pick the best candidate.

    Args:
        word: Word token to check.
        vocab: Known vocabulary set.
        unigram_model: Dict mapping words to unigram probabilities.
        sym_del_dict: Preprocessed symmetric delete dictionary.
        method: "A" (Standard Edit 1) or "B" (Symmetric Delete).

    Returns:
        Best correction string, preserving original capitalization.
    """
    # If punctuation or digit, leave unchanged
    if not word or not any(c.isalpha() for c in word):
        return word

    w_lower = word.lower()
    if w_lower in vocab:
        return word

    # Generate candidate set
    if method == "A":
        all_candidates = generate_candidates_edit1(w_lower)
        valid_candidates = {c for c in all_candidates if c in vocab}
    else:
        valid_candidates = generate_candidates_sym_del(w_lower, sym_del_dict, vocab)

    if not valid_candidates:
        return word

    # Pick candidate with highest unigram probability
    best_candidate = max(valid_candidates, key=lambda c: unigram_model.get(c, 0.0))

    # Preserve capitalization pattern
    if word.isupper():
        return best_candidate.upper()
    elif word[0].isupper():
        return best_candidate.capitalize()
    return best_candidate


def correct_real_word(
    phrase: List[str],
    vocab: Set[str],
    bigram_model: BigramLM,
    sym_del_dict: Dict[str, List[str]],
    threshold: float = 1.0,
) -> List[str]:
    """Corrects real-word errors using context (bigram model probabilities) over edit-distance candidates.

    Compares P(phrase with candidate) vs P(phrase with original).
    If candidate phrase has a significantly higher probability (gain > threshold), suggest correction.

    Args:
        phrase: List of word tokens in sentence.
        vocab: Vocabulary set.
        bigram_model: BigramLM model.
        sym_del_dict: Preprocessed symmetric delete dictionary.
        threshold: Minimum log-likelihood gain required to substitute a word.

    Returns:
        List of corrected words.
    """
    if len(phrase) <= 1:
        return list(phrase)

    corrected = list(phrase)
    n = len(corrected)

    # High-frequency closed-class grammatical tokens that should not be accidentally modified
    protected_words = {
        "a", "i", "the", "in", "on", "at", "to", "me", "it", "is", "be", "as",
        "of", "or", "and", "by", "for", "with", "he", "she", "we", "my", "so",
        "an", "us", "if", "no", "not", "do",
    }

    def clean(t: str) -> str:
        return re.sub(r"[^\w]", "", t).lower()

    for i in range(n):
        orig_token = corrected[i]
        orig_w = clean(orig_token)

        # Skip empty, single letters, out of vocab, or protected closed-class function words
        if not orig_w or len(orig_w) < 2 or orig_w not in vocab or orig_w in protected_words:
            continue

        prev_w = clean(corrected[i - 1]) if i > 0 else "<s>"
        next_w = clean(corrected[i + 1]) if i < n - 1 else "</s>"
        if not prev_w:
            prev_w = "<s>"
        if not next_w:
            next_w = "</s>"

        # Baseline log probability of current word in context
        orig_score = bigram_model.log_prob(prev_w, orig_w) + bigram_model.log_prob(orig_w, next_w)

        candidates = generate_candidates_sym_del(orig_w, sym_del_dict, vocab)

        best_cand = orig_w
        best_gain = 0.0

        for cand in candidates:
            if cand == orig_w:
                continue

            cand_score = bigram_model.log_prob(prev_w, cand) + bigram_model.log_prob(cand, next_w)
            gain = cand_score - orig_score

            if gain > threshold and gain > best_gain:
                best_gain = gain
                best_cand = cand

        if best_cand != orig_w:
            if orig_token.isupper():
                rep = best_cand.upper()
            elif orig_token[0].isupper():
                rep = best_cand.capitalize()
            else:
                rep = best_cand

            trailing_punct = "".join(c for c in orig_token if not c.isalnum())
            corrected[i] = rep + trailing_punct

    return corrected
