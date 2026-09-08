import math
import re
from collections import Counter, defaultdict

# ---------------------------------------------------------
# Q1: Trigram Language Model
# ---------------------------------------------------------
def train_trigram_lm(sents):
    """Train a Trigram LM on a list of [(word, tag)] sentences."""
    uni = Counter()
    bi = Counter()
    tri = Counter()
    for sent in sents:
        words = ["<S>", "<S>"] + [w for w, t in sent] + ["</S>"]
        for w in words:
            uni[w] += 1
        for i in range(len(words) - 1):
            bi[(words[i], words[i+1])] += 1
        for i in range(len(words) - 2):
            tri[(words[i], words[i+1], words[i+2])] += 1
    return uni, bi, tri

def make_log_prob(uni, bi, tri, vocab_size):
    """Return a log-probability function with Add-1 smoothing."""
    def log_prob(w1, w2, w3):
        t = tri.get((w1, w2, w3), 0)
        b = bi.get((w1, w2), 0)
        return math.log((t + 1.0) / (b + vocab_size))
    return log_prob

# ---------------------------------------------------------
# Q1: Viterbi Segmenter
# ---------------------------------------------------------
def viterbi_segment(text, vocab, log_prob_fn, max_word_len=20):
    """
    Segment `text` (unsegmented string) into a list of vocabulary words.
    Uses Viterbi DP with trigram LM scoring.
    """
    n = len(text)
    INF = float("-inf")
    dp = [None] * (n + 1)
    dp[0] = {("<S>", "<S>"): (0.0, None)}

    for i in range(n):
        if dp[i] is None:
            continue
        for j in range(i + 1, min(i + max_word_len + 1, n + 1)):
            word = text[i:j]
            if word not in vocab:
                continue
            if dp[j] is None:
                dp[j] = {}
            for (p2, p1), (score, _) in dp[i].items():
                ns = score + log_prob_fn(p2, p1, word)
                key = (p1, word)
                if key not in dp[j] or ns > dp[j][key][0]:
                    dp[j][key] = (ns, (i, p2))

    if dp[n] is None:
        return []

    best_score, best_key = INF, None
    for (p2, p1), (score, _) in dp[n].items():
        fs = score + log_prob_fn(p2, p1, "</S>")
        if fs > best_score:
            best_score, best_key = fs, (p2, p1)

    if best_key is None:
        return []

    words = []
    ci = n
    p2, p1 = best_key
    while ci > 0:
        words.append(p1)
        _, back = dp[ci][(p2, p1)]
        ci, prev_p2 = back
        p1, p2 = p2, prev_p2
    words.reverse()
    return words

# ---------------------------------------------------------
# Q1: POS Tagging (Second-Order HMM)
# ---------------------------------------------------------
def train_hmm(sents):
    """Train a Second-Order HMM on a list of [(word, tag)] sentences."""
    emission = defaultdict(Counter)
    trans = defaultdict(Counter)
    tag_cnt = Counter()
    tagset = set()

    for sent in sents:
        tags = ["<S>", "<S>"] + [t for _, t in sent] + ["</S>"]
        for w, t in sent:
            emission[t][w] += 1
            tag_cnt[t] += 1
            tagset.add(t)
        for i in range(2, len(tags)):
            trans[(tags[i-2], tags[i-1])][tags[i]] += 1

    return emission, trans, tag_cnt, tagset

def make_hmm_fns(emission, trans, tag_cnt, tagset):
    """Return log-probability functions for emission and transition."""
    vocab_size = sum(len(v) for v in emission.values())
    n_tags = len(tagset)

    def emit_log_prob(tag, word):
        cnt = emission[tag].get(word, 0)
        total = tag_cnt.get(tag, 0)
        return math.log((cnt + 1.0) / (total + vocab_size))

    def trans_log_prob(t1, t2, t3):
        cnt = trans[(t1, t2)].get(t3, 0)
        total = sum(trans[(t1, t2)].values())
        return math.log((cnt + 1.0) / (total + n_tags)) if total > 0 else math.log(1.0 / n_tags)

    return emit_log_prob, trans_log_prob

def viterbi_pos(words, tagset, emit_lp, trans_lp):
    """Second-order Viterbi POS tagger."""
    n = len(words)
    if n == 0:
        return []
    INF = float("-inf")
    dp = [{} for _ in range(n + 1)]
    dp[0][("<S>", "<S>")] = (0.0, None)
    tags = list(tagset)

    for i, word in enumerate(words):
        for (t1, t2), (score, _) in dp[i].items():
            for t3 in tags:
                ns = score + emit_lp(t3, word) + trans_lp(t1, t2, t3)
                key = (t2, t3)
                if key not in dp[i+1] or ns > dp[i+1][key][0]:
                    dp[i+1][key] = (ns, t1)

    best_score, best_key = INF, None
    for (t1, t2), (score, _) in dp[n].items():
        fs = score + trans_lp(t1, t2, "</S>")
        if fs > best_score:
            best_score, best_key = fs, (t1, t2)

    if best_key is None:
        return ["NOUN"] * n

    pred = []
    ci = n
    t1, t2 = best_key
    while ci > 0:
        pred.append(t2)
        _, prev_t1 = dp[ci][(t1, t2)]
        ci -= 1
        t1, t2 = prev_t1, t1
    pred.reverse()
    return pred
