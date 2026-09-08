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
        words = ["<S>", "<S>"] + [w.lower() for w, _ in sent] + ["</S>"]
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
# Q1: POS Tagging (Feature-Based Classifier + Beam Decoder)
# ---------------------------------------------------------

def extract_morphological_features(word: str):
    """Extracts morphological and orthographic features from a word.
    Includes suffixes (1-4 chars), prefixes (2-3 chars), and shape features.
    """
    features = []
    w_lower = word.lower()
    w_len = len(word)

    for slen in (1, 2, 3, 4):
        if w_len > slen:
            features.append(f"suf_{w_lower[-slen:]}")

    for plen in (2, 3):
        if w_len > plen + 1:
            features.append(f"pref_{w_lower[:plen]}")

    if any(c.isdigit() for c in word):
        features.append("shape_has_digit")
    if "-" in word:
        features.append("shape_has_hyphen")
    if word.isupper() and w_len > 1:
        features.append("shape_all_caps")
    elif word[0].isupper() and (w_len == 1 or word[1:].islower()):
        features.append("shape_capitalized")
    elif word.islower():
        features.append("shape_all_lower")

    return features


class MorphologyFeatureModel:
    """Subword morphological emission estimator for unseen/rare words."""

    def __init__(self, feature_tag_counts, tag_feature_totals, all_features, beta=0.1):
        self.feature_tag_counts = feature_tag_counts
        self.tag_feature_totals = tag_feature_totals
        self.all_features = all_features
        self.vocab_f_size = max(len(all_features), 1)
        self.beta = beta

    def log_prob(self, word: str, tag: str) -> float:
        """Computes log P(morphology(word) | tag) with smoothed naive bayes."""
        feats = extract_morphological_features(word)
        if not feats:
            return -12.0

        total_f_for_tag = self.tag_feature_totals.get(tag, 0)
        denom = total_f_for_tag + self.beta * self.vocab_f_size
        log_prob = 0.0

        for f in feats:
            c = self.feature_tag_counts[f].get(tag, 0)
            p = (c + self.beta) / denom
            log_prob += math.log(p)

        return log_prob / math.sqrt(len(feats))


def train_hmm(sents):
    """Train a Second-Order HMM and Feature Morphology Model on a list of [(word, tag)] sentences."""
    emission = defaultdict(Counter)
    trans = defaultdict(Counter)
    tag_cnt = Counter()
    tagset = set()
    word_freq = Counter()
    feat_tag_cnt = defaultdict(Counter)
    tag_feat_total = Counter()
    all_feats = set()

    for sent in sents:
        tags = ["<S>", "<S>"] + [t for _, t in sent] + ["</S>"]
        for w, t in sent:
            w_lower = w.lower()
            emission[t][w_lower] += 1
            tag_cnt[t] += 1
            tagset.add(t)
            word_freq[w_lower] += 1
        for i in range(2, len(tags)):
            trans[(tags[i-2], tags[i-1])][tags[i]] += 1

    for sent in sents:
        for w, t in sent:
            weight = 3 if word_freq[w.lower()] <= 5 else 1
            feats = extract_morphological_features(w)
            for f in feats:
                feat_tag_cnt[f][t] += weight
                tag_feat_total[t] += weight
                all_feats.add(f)

    morph_model = MorphologyFeatureModel(feat_tag_cnt, tag_feat_total, all_feats)
    return emission, trans, tag_cnt, tagset, morph_model


def make_hmm_fns(emission, trans, tag_cnt, tagset, morph_model=None):
    """Return log-probability functions for feature-aware emission and transition."""
    vocab_size = sum(len(v) for v in emission.values())
    n_tags = len(tagset)

    def emit_log_prob(tag, word):
        w_lower = word.lower()
        cnt = emission[tag].get(w_lower, 0)
        if cnt > 0:
            total = tag_cnt.get(tag, 0)
            return math.log((cnt + 1.0) / (total + vocab_size))
        elif morph_model is not None:
            return morph_model.log_prob(word, tag)
        else:
            total = tag_cnt.get(tag, 0)
            return math.log(1.0 / (total + vocab_size))

    def trans_log_prob(t1, t2, t3):
        cnt = trans[(t1, t2)].get(t3, 0)
        total = sum(trans[(t1, t2)].values())
        return math.log((cnt + 1.0) / (total + n_tags)) if total > 0 else math.log(1.0 / n_tags)

    return emit_log_prob, trans_log_prob


def beam_pos_tagger(words, tagset, emit_lp, trans_lp, beam_width=5):
    """Beam search decoder for POS tagging using feature-based emission and 2nd-order transitions."""
    n = len(words)
    if n == 0:
        return []

    tags = list(tagset)
    # Each beam item is (score, [t_prev2, t_prev1], [predicted_tags])
    beam = [(0.0, ["<S>", "<S>"], [])]

    for i, word in enumerate(words):
        candidates = []
        for score, (t1, t2), history in beam:
            for t3 in tags:
                ns = score + emit_lp(t3, word) + trans_lp(t1, t2, t3)
                candidates.append((ns, [t2, t3], history + [t3]))

        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = candidates[:beam_width]

    best_score = float("-inf")
    best_tags = ["NOUN"] * n

    for score, (t1, t2), history in beam:
        final_score = score + trans_lp(t1, t2, "</S>")
        if final_score > best_score:
            best_score = final_score
            best_tags = history

    return best_tags


def viterbi_pos(words, tagset, emit_lp, trans_lp):
    """Second-order Viterbi POS tagger (retained for baseline comparison)."""
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


def joint_beam_segment_token(token, vocab, log_prob_fn, tagset, emit_lp, trans_lp,
                             alpha=1.0, beta=0.5, beam_width=10, max_word_len=20):
    """Joint beam-search decoder for a single token using LM (alpha) and POS (beta).
    Returns (split_words, predicted_tags, did_split).
    """
    clean = "".join(c for c in token if c.isalnum()).lower()
    if not clean:
        return [token], ["NOUN"], False

    n = len(clean)
    # beam[j] contains list of (score, (w_prev2, w_prev1), (t_prev2, t_prev1), words_list, tags_list)
    beams = [[] for _ in range(n + 1)]
    beams[0] = [(0.0, ("<S>", "<S>"), ("<S>", "<S>"), [], [])]
    tags = list(tagset)

    for i in range(n):
        if not beams[i]:
            continue
        limit = min(n, i + max_word_len)
        for j in range(i + 1, limit + 1):
            sub = clean[i:j]
            if sub not in vocab:
                continue
            for score, (w1, w2), (t1, t2), w_hist, t_hist in beams[i]:
                lm_lp = log_prob_fn(w1, w2, sub)
                # Find best POS tag for sub in this context
                best_tag = "NOUN"
                best_tag_lp = float("-inf")
                for t3 in tags:
                    pos_lp = emit_lp(t3, sub) + trans_lp(t1, t2, t3)
                    if pos_lp > best_tag_lp:
                        best_tag_lp = pos_lp
                        best_tag = t3

                step_score = score + alpha * lm_lp + beta * best_tag_lp
                beams[j].append((step_score, (w2, sub), (t2, best_tag), w_hist + [sub], t_hist + [best_tag]))

        if beams[j]:
            beams[j].sort(key=lambda x: x[0], reverse=True)
            beams[j] = beams[j][:beam_width]

    if not beams[n]:
        # Fallback to single word
        tag = "NOUN"
        if clean in vocab:
            tag = max(tags, key=lambda t: emit_lp(t, clean))
        return [clean], [tag], False

    best_cand = max(beams[n], key=lambda x: x[0])
    splits, split_tags = best_cand[3], best_cand[4]

    # Validate split
    if (len(splits) >= 2
            and all((len(w) >= 2 or w in {"a", "i"}) and w in vocab for w in splits)):
        return splits, split_tags, True

    # Otherwise treat as single word
    tag = "NOUN"
    if clean in vocab:
        tag = max(tags, key=lambda t: emit_lp(t, clean))
    return [clean], [tag], False

