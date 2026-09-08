"""Word Segmentation using Trigram Language Model and Viterbi Decoding."""

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


class TrigramLM:
    """A smoothed Trigram Language Model for word scoring and segmentation."""

    def __init__(
        self,
        unigram_counts: Counter,
        bigram_counts: Counter,
        trigram_counts: Counter,
        total_tokens: int,
        vocabulary: Set[str],
        lambdas: Tuple[float, float, float, float] = (0.60, 0.25, 0.14, 0.01),
        max_word_len: int = 25,
    ):
        self.unigrams = unigram_counts
        self.bigrams = bigram_counts
        self.trigrams = trigram_counts
        self.total_tokens = max(total_tokens, 1)
        self.vocab = vocabulary
        self.vocab_lower = {w.lower() for w in vocabulary}
        self.vocab_size = max(len(vocabulary), 1)
        self.l3, self.l2, self.l1, self.l0 = lambdas
        self.max_word_len = max_word_len
        self.min_log_prob = -50.0

        # Precompute bigram contexts for normalization
        self.bigram_context_counts = Counter()
        for (w1, w2), c in self.bigrams.items():
            self.bigram_context_counts[(w1, w2)] += c

        self.unigram_context_counts = Counter()
        for (w1, _), c in self.bigrams.items():
            self.unigram_context_counts[w1] += c

    def score(self, w3: str, w1: str = "<s>", w2: str = "<s>") -> float:
        """Returns the smoothed log probability log P(w3 | w1, w2)."""
        w3_clean = w3.lower()
        w1_clean = w1.lower()
        w2_clean = w2.lower()

        # Trigram MLE
        c_tri = self.trigrams.get((w1_clean, w2_clean, w3_clean), 0)
        c_bi_ctx = self.bigram_context_counts.get((w1_clean, w2_clean), 0)
        p_tri = (c_tri / c_bi_ctx) if c_bi_ctx > 0 else 0.0

        # Bigram MLE
        c_bi = self.bigrams.get((w2_clean, w3_clean), 0)
        c_uni_ctx = self.unigram_context_counts.get(w2_clean, 0)
        p_bi = (c_bi / c_uni_ctx) if c_uni_ctx > 0 else 0.0

        # Unigram MLE
        c_uni = self.unigrams.get(w3_clean, 0)
        p_uni = c_uni / self.total_tokens if self.total_tokens > 0 else 0.0

        # Uniform / OOV fallback
        p_uniform = 1.0 / self.vocab_size

        prob = (
            self.l3 * p_tri
            + self.l2 * p_bi
            + self.l1 * p_uni
            + self.l0 * p_uniform
        )

        if prob <= 0.0:
            return self.min_log_prob

        log_p = math.log(prob)

        # Apply a length penalty for out-of-vocabulary words
        if w3_clean not in self.vocab_lower and w3 not in ("<s>", "</s>"):
            # Strong penalty for single unknown characters, mild penalty for longer unknown words
            log_p -= 3.0 + 1.2 * len(w3)

        return log_p

    def contains(self, word: str) -> bool:
        """Checks if word or its lowercased version is in the vocabulary."""
        return word in self.vocab or word.lower() in self.vocab_lower


def train_trigram_lm(
    corpus: List[List[str]],
    lambdas: Tuple[float, float, float, float] = (0.60, 0.25, 0.14, 0.01),
) -> TrigramLM:
    """Trains a trigram language model on the given corpus.

    Args:
        corpus: A list of sentences, where each sentence is a list of string tokens.
        lambdas: Interpolation weights (lambda_trigram, lambda_bigram, lambda_unigram, lambda_uniform).

    Returns:
        A trained TrigramLM instance.
    """
    unigrams: Counter = Counter()
    bigrams: Counter = Counter()
    trigrams: Counter = Counter()
    vocab: Set[str] = set()
    total_tokens = 0
    max_len = 1

    for sentence in corpus:
        if not sentence:
            continue
        cleaned = [token.strip() for token in sentence if token.strip()]
        if not cleaned:
            continue

        for token in cleaned:
            vocab.add(token)
            if len(token) > max_len:
                max_len = len(token)

        # Lowercase for language modeling counts
        padded = ["<s>", "<s>"] + [t.lower() for t in cleaned] + ["</s>"]

        for i in range(2, len(padded)):
            w3 = padded[i]
            w2 = padded[i - 1]
            w1 = padded[i - 2]

            trigrams[(w1, w2, w3)] += 1
            bigrams[(w2, w3)] += 1
            unigrams[w3] += 1
            total_tokens += 1

    # Cap max_word_len to a reasonable threshold (e.g. 25)
    max_word_len = min(max_len, 25)

    return TrigramLM(
        unigram_counts=unigrams,
        bigram_counts=bigrams,
        trigram_counts=trigrams,
        total_tokens=total_tokens,
        vocabulary=vocab,
        lambdas=lambdas,
        max_word_len=max_word_len,
    )


def viterbi_segmentation(
    text: str,
    lm_model: Any,
    beam_width: int = 40,
) -> List[str]:
    """Segments a continuous string of text into words using the Viterbi algorithm and a trigram LM.

    Args:
        text: Continuous unsegmented string of characters.
        lm_model: Trained TrigramLM instance.
        beam_width: Maximum number of hypotheses to retain per character index.

    Returns:
        A list of segmented word strings.
    """
    if not text:
        return []

    # Strip any whitespace
    raw_text = "".join(text.split())
    n = len(raw_text)
    if n == 0:
        return []

    max_word_len = getattr(lm_model, "max_word_len", 25)

    # dp[j] maps state (u, v) -> (log_score, (prev_idx, prev_u))
    # where v ends at j, and u ends at prev_idx
    dp: List[Dict[Tuple[str, str], Tuple[float, Optional[Tuple[int, str]]]]] = [
        {} for _ in range(n + 1)
    ]
    dp[0][("<s>", "<s>")] = (0.0, None)

    for i in range(n):
        if not dp[i]:
            continue

        # Prune states at position i to beam_width
        if len(dp[i]) > beam_width:
            sorted_states = sorted(dp[i].items(), key=lambda item: item[1][0], reverse=True)[
                :beam_width
            ]
            dp[i] = dict(sorted_states)

        # Discover valid candidates starting at i
        limit = min(n, i + max_word_len)
        has_vocab_match = False

        candidates: List[str] = []
        for j in range(i + 1, limit + 1):
            sub = raw_text[i:j]
            if hasattr(lm_model, "contains") and lm_model.contains(sub):
                candidates.append(sub)
                has_vocab_match = True

        # If no vocabulary word matches from this index, or to allow single-char fallback
        if not has_vocab_match:
            candidates.append(raw_text[i : i + 1])
        elif len(raw_text[i : i + 1]) == 1 and raw_text[i : i + 1] not in candidates:
            # Also allow single-char fallback even if longer match exists
            candidates.append(raw_text[i : i + 1])

        for w in candidates:
            j = i + len(w)
            for (u, v), (prev_score, _) in dp[i].items():
                trans_score = lm_model.score(w, u, v)
                new_score = prev_score + trans_score
                new_state = (v, w)

                existing = dp[j].get(new_state)
                if existing is None or new_score > existing[0]:
                    dp[j][new_state] = (new_score, (i, u))

    if not dp[n]:
        # Fallback to greedy baseline if no complete path was formed
        vocab = getattr(lm_model, "vocab", set())
        return greedy_longest_match_baseline(raw_text, vocab)

    # Transition to end-of-sentence marker </s>
    best_final_score = -float("inf")
    best_final_state: Optional[Tuple[str, str]] = None

    for (u, v), (score, _) in dp[n].items():
        end_score = score + lm_model.score("</s>", u, v)
        if end_score > best_final_score:
            best_final_score = end_score
            best_final_state = (u, v)

    if best_final_state is None:
        best_final_state = max(dp[n].items(), key=lambda item: item[1][0])[0]

    # Traceback to extract words
    segmented_words: List[str] = []
    curr_j = n
    curr_state = best_final_state

    while curr_j > 0 and curr_state is not None:
        u, v = curr_state
        segmented_words.append(v)
        entry = dp[curr_j].get(curr_state)
        if entry is None or entry[1] is None:
            break
        prev_idx, prev_u = entry[1]
        curr_state = (prev_u, u)
        curr_j = prev_idx

    segmented_words.reverse()
    return segmented_words


def greedy_longest_match_baseline(text: str, vocabulary: Set[str]) -> List[str]:
    """Baseline for segmentation: always matches the longest word from vocabulary at each point.

    Args:
        text: Continuous unsegmented string of characters.
        vocabulary: Set of known words.

    Returns:
        List of segmented word strings.
    """
    if not text:
        return []

    raw_text = "".join(text.split())
    n = len(raw_text)
    if n == 0:
        return []

    vocab_lower = {w.lower() for w in vocabulary}
    max_len = max((len(w) for w in vocabulary), default=25)

    words: List[str] = []
    i = 0

    while i < n:
        matched = False
        upper = min(n, i + max_len)
        for j in range(upper, i, -1):
            sub = raw_text[i:j]
            if sub in vocabulary or sub.lower() in vocab_lower:
                words.append(sub)
                i = j
                matched = True
                break

        if not matched:
            words.append(raw_text[i : i + 1])
            i += 1

    return words
