import math

# ---------------------------------------------------------
# Q3: Spelling Corrector Core Functions
# ---------------------------------------------------------

BIGRAM_K = 0.01

def damerau_levenshtein(s1: str, s2: str) -> int:
    """Computes the Damerau-Levenshtein distance between two strings."""
    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1,      # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )
            # Transposition
            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + cost)

    return dp[len1][len2]


def _generate_edit1_strings(word):
    """Generate all strings reachable by one edit."""
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    w = word.lower()
    n = len(w)

    deletes    = [w[:i] + w[i+1:] for i in range(n)]
    transposes = [w[:i] + w[i+1] + w[i] + w[i+2:] for i in range(n - 1)]
    replaces   = [w[:i] + c + w[i+1:] for i in range(n) for c in alphabet]
    inserts    = [w[:i] + c + w[i:] for i in range(n + 1) for c in alphabet]

    return set(deletes + transposes + replaces + inserts)


def method_a_candidates(word, vocab):
    """Method A — Edit Distance 1 with DL Verification."""
    word = word.lower()
    raw_candidates = _generate_edit1_strings(word)
    vocab_candidates = raw_candidates & vocab
    return {c for c in vocab_candidates if damerau_levenshtein(word, c) <= 1}


def build_delete_index(vocab):
    """Build the Symmetric Delete index."""
    delete_index = {}
    for word in vocab:
        for i in range(len(word)):
            deleted = word[:i] + word[i + 1:]
            if deleted not in delete_index:
                delete_index[deleted] = set()
            delete_index[deleted].add(word)
    return delete_index


def method_b_candidates(word, delete_index):
    """Method B — Symmetric Delete candidate generation."""
    word = word.lower()
    candidates = set()
    candidates.update(delete_index.get(word, set()))
    for i in range(len(word)):
        deleted = word[:i] + word[i + 1:]
        candidates.update(delete_index.get(deleted, set()))
    return candidates


def best_unigram_candidate(candidates, word_counts):
    """Pick the best candidate based on unigram frequency."""
    if not candidates:
        return None
    return max(candidates, key=lambda word: word_counts.get(word, 0))


def best_candidate_with_context(word, candidates, uni_counts, prev_word=None, bi_by_prev=None):
    """Pick best non-word candidate using local bigram context if available, with prefix bonus."""
    if not candidates:
        return word

    if prev_word and bi_by_prev:
        prev_clean = prev_word.lower()
        if prev_clean in bi_by_prev:
            def score(cand):
                bi_cnt = bi_by_prev[prev_clean].get(cand, 0)
                uni_cnt = uni_counts.get(cand, 0)
                # If candidate extends/shares word prefix, prioritize it as typing omission
                prefix_bonus = 2.5 if cand.startswith(word) else 1.0
                return (bi_cnt * 1000 + uni_cnt) * prefix_bonus

            return max(candidates, key=score)

    return max(candidates, key=lambda w: uni_counts.get(w, 0))


def bigram_probability(previous, word, bigram_counts, unigram_context_counts, vocab_size):
    """Returns the add-k smoothed bigram probability P(word | previous)."""
    c_bigram  = bigram_counts[previous].get(word, 0)
    c_context = unigram_context_counts.get(previous, 0)
    return (c_bigram + BIGRAM_K) / (c_context + BIGRAM_K * vocab_size)


def real_word_context_score(previous_word, word, next_word, bigram_counts, unigram_context_counts, vocab_size):
    """Score a word using left and right bigram log-probabilities."""
    left_log_prob  = math.log(bigram_probability(previous_word, word, bigram_counts, unigram_context_counts, vocab_size))
    right_log_prob = math.log(bigram_probability(word, next_word, bigram_counts, unigram_context_counts, vocab_size))
    return left_log_prob + right_log_prob


def correct_real_word(word, previous_word, next_word, delete_index, bigram_counts, unigram_context_counts, vocab_size, improvement_threshold=2.5, method="B", vocab=None):
    """Correct a real-word spelling error using local bigram context."""
    if method == "A" and vocab is not None:
        candidates = method_a_candidates(word, vocab)
    else:
        candidates = method_b_candidates(word, delete_index)

    if not candidates:
        return word

    # Filter candidates: must be true edit-distance-1 (DL <= 1) and have observed bigram support
    filtered = [
        c for c in candidates
        if damerau_levenshtein(word, c) <= 1
        and (bigram_counts[previous_word].get(c, 0) > 0 or bigram_counts[c].get(next_word, 0) > 0)
    ]
    if not filtered:
        return word

    original_score = real_word_context_score(previous_word, word, next_word, bigram_counts, unigram_context_counts, vocab_size)
    best_word  = word
    best_score = original_score

    COMMON_CONFUSIONS = {
        ("meat", "meet"), ("meet", "meat"),
        ("peace", "piece"), ("piece", "peace"),
        ("hear", "here"), ("here", "hear"),
        ("there", "their"), ("their", "there"),
        ("weather", "whether"), ("whether", "weather"),
        ("principal", "principle"), ("principle", "principal"),
        ("loose", "lose"), ("lose", "loose"),
    }

    for candidate in filtered:
        candidate_score = real_word_context_score(previous_word, candidate, next_word, bigram_counts, unigram_context_counts, vocab_size)
        if (word, candidate) in COMMON_CONFUSIONS:
            candidate_score += 2.0
        if candidate_score > best_score:
            best_word  = candidate
            best_score = candidate_score

    if (best_score - original_score) >= improvement_threshold:
        return best_word

    return word
