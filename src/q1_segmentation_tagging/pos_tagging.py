"""Morphology-Aware Hidden Markov Model (HMM) Part-of-Speech Tagger with Viterbi Decoding."""

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


def extract_morphological_features(word: str) -> List[str]:
    """Extracts morphological and orthographic features from a word.

    Features include suffixes (lengths 1-4), prefixes (lengths 2-3),
    and shape properties (capitalization, digits, hyphens).
    """
    features: List[str] = []
    w_lower = word.lower()
    w_len = len(word)

    # Suffixes
    for slen in (1, 2, 3, 4):
        if w_len > slen:
            features.append(f"suf_{w_lower[-slen:]}")

    # Prefixes
    for plen in (2, 3):
        if w_len > plen + 1:
            features.append(f"pref_{w_lower[:plen]}")

    # Orthographic / Shape features
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


class MorphologyModel:
    """Subword morphological emission estimator for unseen/rare words."""

    def __init__(
        self,
        feature_tag_counts: Dict[str, Counter],
        tag_feature_totals: Counter,
        all_features: Set[str],
        beta: float = 0.1,
    ):
        self.feature_tag_counts = feature_tag_counts
        self.tag_feature_totals = tag_feature_totals
        self.all_features = all_features
        self.vocab_f_size = max(len(all_features), 1)
        self.beta = beta

    def log_prob(self, word: str, tag: str) -> float:
        """Computes log P(morphology(word) | tag) with smoothed naive bayes."""
        features = extract_morphological_features(word)
        if not features:
            return -10.0

        total_f_for_tag = self.tag_feature_totals.get(tag, 0)
        denom = total_f_for_tag + self.beta * self.vocab_f_size
        log_prob = 0.0

        for f in features:
            c = self.feature_tag_counts[f][tag] if f in self.feature_tag_counts else 0
            p = (c + self.beta) / denom
            log_prob += math.log(p)

        # Normalize by feature length to prevent over-penalizing words with many features
        return log_prob / math.sqrt(len(features))


def train_pos_tagger(
    corpus: List[List[Tuple[str, str]]],
    alpha: float = 0.1,
    rare_threshold: int = 5,
) -> Dict[str, Any]:
    """Trains emission and transition probabilities for POS tagging with morphology awareness.

    Args:
        corpus: Training sentences, where each sentence is a list of (word, tag) tuples.
        alpha: Smoothing parameter for transition probabilities.
        rare_threshold: Words with frequency <= rare_threshold are used to train the morphology model.

    Returns:
        A dictionary containing all tagger model components:
        'tags', 'transitions', 'emissions', 'tag_counts', 'morphology_model',
        'word_vocab', 'start_transitions', 'end_transitions'.
    """
    tag_counts: Counter = Counter()
    word_counts: Counter = Counter()
    word_tag_counts: Dict[str, Counter] = defaultdict(Counter)
    tag_word_counts: Dict[str, Counter] = defaultdict(Counter)

    transition_counts: Dict[str, Counter] = defaultdict(Counter)
    start_counts: Counter = Counter()
    end_counts: Counter = Counter()

    tags: Set[str] = set()

    for sentence in corpus:
        if not sentence:
            continue

        prev_tag = "<s>"
        for i, (word, tag) in enumerate(sentence):
            tags.add(tag)
            tag_counts[tag] += 1
            word_counts[word.lower()] += 1
            word_tag_counts[word.lower()][tag] += 1
            tag_word_counts[tag][word.lower()] += 1

            if i == 0:
                start_counts[tag] += 1
            else:
                transition_counts[prev_tag][tag] += 1
            prev_tag = tag

        end_counts[prev_tag] += 1

    tag_list = sorted(list(tags))
    num_tags = len(tag_list)
    num_sentences = max(len(corpus), 1)

    # Train Morphology Model using rare & all words
    feature_tag_counts: Dict[str, Counter] = defaultdict(Counter)
    tag_feature_totals: Counter = Counter()
    all_features: Set[str] = set()

    for sentence in corpus:
        for word, tag in sentence:
            # Emphasize rare words for morphology model, but also include common words
            w_count = word_counts[word.lower()]
            weight = 3 if w_count <= rare_threshold else 1

            feats = extract_morphological_features(word)
            for f in feats:
                feature_tag_counts[f][tag] += weight
                tag_feature_totals[tag] += weight
                all_features.add(f)

    morph_model = MorphologyModel(
        feature_tag_counts=feature_tag_counts,
        tag_feature_totals=tag_feature_totals,
        all_features=all_features,
    )

    # Precompute smoothed transition log-probabilities
    # log P(tag_j | tag_i)
    transitions: Dict[Tuple[str, str], float] = {}
    for t_prev in tag_list:
        denom = tag_counts[t_prev] + alpha * num_tags
        for t_curr in tag_list:
            c = transition_counts[t_prev][t_curr]
            prob = (c + alpha) / denom
            transitions[(t_prev, t_curr)] = math.log(prob)

    # Start transitions log P(t | <s>)
    start_transitions: Dict[str, float] = {}
    start_denom = num_sentences + alpha * num_tags
    for t in tag_list:
        c = start_counts[t]
        start_transitions[t] = math.log((c + alpha) / start_denom)

    # End transitions log P(</s> | t)
    end_transitions: Dict[str, float] = {}
    for t in tag_list:
        denom = tag_counts[t] + alpha * 2
        c = end_counts[t]
        end_transitions[t] = math.log((c + alpha) / denom)

    # Precompute lexical emission log-probabilities for known words: log P(w | t)
    # P(w | t) = C(t, w) / C(t)
    emissions: Dict[Tuple[str, str], float] = {}
    for tag in tag_list:
        c_tag = tag_counts[tag]
        if c_tag > 0:
            for w, c_tw in tag_word_counts[tag].items():
                emissions[(w, tag)] = math.log(c_tw / c_tag)

    return {
        "tags": tag_list,
        "transitions": transitions,
        "start_transitions": start_transitions,
        "end_transitions": end_transitions,
        "emissions": emissions,
        "tag_counts": tag_counts,
        "morphology_model": morph_model,
        "word_vocab": set(word_counts.keys()),
    }


def viterbi_pos_tagging(
    words: List[str],
    tagger_model: Dict[str, Any],
) -> List[Tuple[str, str]]:
    """Tags a sequence of words using Viterbi decoding and trained emission/transition probabilities.

    Args:
        words: List of word strings to tag.
        tagger_model: Model dictionary returned by train_pos_tagger.

    Returns:
        List of (word, predicted_tag) tuples.
    """
    if not words:
        return []

    tags: List[str] = tagger_model["tags"]
    transitions: Dict[Tuple[str, str], float] = tagger_model["transitions"]
    start_transitions: Dict[str, float] = tagger_model["start_transitions"]
    end_transitions: Dict[str, float] = tagger_model["end_transitions"]
    emissions: Dict[Tuple[str, str], float] = tagger_model["emissions"]
    morph_model: MorphologyModel = tagger_model["morphology_model"]
    word_vocab: Set[str] = tagger_model["word_vocab"]

    n = len(words)

    # Helper to calculate log P(word | tag)
    def get_emission_log_prob(w: str, t: str) -> float:
        w_lower = w.lower()
        if w_lower in word_vocab:
            # Known word
            pair = (w_lower, t)
            if pair in emissions:
                # Small interpolation with morphology for robustness
                return emissions[pair]
            else:
                # Word was seen, but never with this tag
                return -25.0
        else:
            # Out of vocabulary: use morphology model
            return morph_model.log_prob(w, t)

    # Viterbi tables
    # viterbi[i][t] = best log probability ending at step i with tag t
    # backpointer[i][t] = previous tag that achieved this max
    viterbi: List[Dict[str, float]] = [{} for _ in range(n)]
    backpointer: List[Dict[str, str]] = [{} for _ in range(n)]

    # Step 0: Initialization
    w0 = words[0]
    for t in tags:
        start_p = start_transitions.get(t, -20.0)
        emit_p = get_emission_log_prob(w0, t)
        viterbi[0][t] = start_p + emit_p

    # Steps 1 to n-1: Recursion
    for i in range(1, n):
        wi = words[i]
        for t_curr in tags:
            emit_p = get_emission_log_prob(wi, t_curr)
            best_prev_score = -float("inf")
            best_prev_tag = tags[0]

            for t_prev in tags:
                trans_p = transitions.get((t_prev, t_curr), -25.0)
                score = viterbi[i - 1][t_prev] + trans_p
                if score > best_prev_score:
                    best_prev_score = score
                    best_prev_tag = t_prev

            viterbi[i][t_curr] = best_prev_score + emit_p
            backpointer[i][t_curr] = best_prev_tag

    # Termination: step n with </s>
    best_final_score = -float("inf")
    best_last_tag = tags[0]

    for t in tags:
        end_p = end_transitions.get(t, -20.0)
        total_score = viterbi[n - 1][t] + end_p
        if total_score > best_final_score:
            best_final_score = total_score
            best_last_tag = t

    # Traceback
    predicted_tags: List[str] = [best_last_tag]
    curr_tag = best_last_tag
    for i in range(n - 1, 0, -1):
        prev_tag = backpointer[i][curr_tag]
        predicted_tags.append(prev_tag)
        curr_tag = prev_tag

    predicted_tags.reverse()
    return list(zip(words, predicted_tags))


def most_frequent_tag_baseline(
    words: List[str],
    training_data: List[List[Tuple[str, str]]],
) -> List[Tuple[str, str]]:
    """Baseline for POS tagging: assigns the most frequent tag observed for each word in training data.

    If a word was never observed in training data, assigns the global majority tag.

    Args:
        words: List of word strings to tag.
        training_data: Annotated sentences used to compute word-tag frequencies.

    Returns:
        List of (word, predicted_tag) tuples.
    """
    if not words:
        return []

    word_tag_counts: Dict[str, Counter] = defaultdict(Counter)
    global_tag_counts: Counter = Counter()

    for sentence in training_data:
        for word, tag in sentence:
            word_tag_counts[word.lower()][tag] += 1
            global_tag_counts[tag] += 1

    # Overall most frequent tag across the corpus
    default_tag = global_tag_counts.most_common(1)[0][0] if global_tag_counts else "NOUN"

    # Precompute most frequent tag for each observed word
    word_to_mft: Dict[str, str] = {}
    for w, counts in word_tag_counts.items():
        word_to_mft[w] = counts.most_common(1)[0][0]

    result: List[Tuple[str, str]] = []
    for w in words:
        tag = word_to_mft.get(w.lower(), default_tag)
        result.append((w, tag))

    return result
