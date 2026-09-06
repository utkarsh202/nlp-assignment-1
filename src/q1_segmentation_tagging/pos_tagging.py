"""Morphology-Aware Trigram HMM Part-of-Speech Tagger with Viterbi Decoding."""

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple


def extract_morphological_features(word: str) -> List[str]:
    """Extract morphological and orthographic features from a word."""
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

    # Shape features
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
        """Compute log P(morphology(word) | tag)."""
        features = extract_morphological_features(word)

        if not features:
            return -10.0

        total_f_for_tag = self.tag_feature_totals.get(tag, 0)
        denom = total_f_for_tag + self.beta * self.vocab_f_size

        log_prob = 0.0

        for feature in features:
            count = self.feature_tag_counts[feature].get(tag, 0)
            probability = (count + self.beta) / denom
            log_prob += math.log(probability)

        return log_prob / math.sqrt(len(features))


def train_pos_tagger(
    corpus: List[List[Tuple[str, str]]],
    alpha: float = 0.1,
    rare_threshold: int = 5,
) -> Dict[str, Any]:
    """Train a morphology-aware trigram POS HMM.

    Transition probability:

        P(tag_i | tag_i-2, tag_i-1)
    """

    tag_counts: Counter = Counter()
    word_counts: Counter = Counter()

    word_tag_counts: Dict[str, Counter] = defaultdict(Counter)
    tag_word_counts: Dict[str, Counter] = defaultdict(Counter)

    # Context = (previous_previous_tag, previous_tag)
    # Value = counts of possible current tags.
    transition_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)

    start_counts: Counter = Counter()
    end_counts: Counter = Counter()

    tags: Set[str] = set()
    num_sentences = 0

    # ---------------------------------------------------------
    # Collect corpus counts
    # ---------------------------------------------------------

    for sentence in corpus:
        if not sentence:
            continue

        num_sentences += 1

        prev_prev_tag = "<s>"
        prev_tag = "<s>"

        for i, (word, tag) in enumerate(sentence):
            tags.add(tag)

            tag_counts[tag] += 1
            word_counts[word.lower()] += 1
            word_tag_counts[word.lower()][tag] += 1
            tag_word_counts[tag][word.lower()] += 1

            if i == 0:
                start_counts[tag] += 1

            # Required trigram transition:
            # P(current_tag | previous_two_tags)
            transition_counts[(prev_prev_tag, prev_tag)][tag] += 1

            prev_prev_tag = prev_tag
            prev_tag = tag

        end_counts[prev_tag] += 1

    tag_list = sorted(tags)
    num_tags = len(tag_list)
    num_sentences = max(num_sentences, 1)

    # ---------------------------------------------------------
    # Morphology model
    # ---------------------------------------------------------

    feature_tag_counts: Dict[str, Counter] = defaultdict(Counter)
    tag_feature_totals: Counter = Counter()
    all_features: Set[str] = set()

    for sentence in corpus:
        for word, tag in sentence:
            count = word_counts[word.lower()]

            # Give rare words more influence.
            weight = 3 if count <= rare_threshold else 1

            features = extract_morphological_features(word)

            for feature in features:
                feature_tag_counts[feature][tag] += weight
                tag_feature_totals[tag] += weight
                all_features.add(feature)

    morph_model = MorphologyModel(
        feature_tag_counts=feature_tag_counts,
        tag_feature_totals=tag_feature_totals,
        all_features=all_features,
    )

    # ---------------------------------------------------------
    # Store trigram transition probabilities.
    #
    # Only observed contexts are stored.
    # ---------------------------------------------------------

    transitions: Dict[Tuple[str, str, str], float] = {}

    for context, next_counts in transition_counts.items():
        context_total = sum(next_counts.values())

        denominator = context_total + alpha * num_tags

        prev_prev_tag, prev_tag = context

        for current_tag, count in next_counts.items():
            probability = (count + alpha) / denominator

            transitions[
                (prev_prev_tag, prev_tag, current_tag)
            ] = math.log(probability)

    # Log probability for an unseen transition.
    unseen_transition_log_prob = math.log(
        alpha / (alpha * num_tags)
    )

    # ---------------------------------------------------------
    # Start probabilities
    # ---------------------------------------------------------

    start_transitions: Dict[str, float] = {}

    start_denominator = num_sentences + alpha * num_tags

    for tag in tag_list:
        count = start_counts[tag]

        start_transitions[tag] = math.log(
            (count + alpha) / start_denominator
        )

    # ---------------------------------------------------------
    # End probabilities
    # ---------------------------------------------------------

    end_transitions: Dict[str, float] = {}

    for tag in tag_list:
        denominator = tag_counts[tag] + alpha * 2
        count = end_counts[tag]

        end_transitions[tag] = math.log(
            (count + alpha) / denominator
        )

    # ---------------------------------------------------------
    # Lexical emissions
    #
    # P(word | tag)
    # ---------------------------------------------------------

    emissions: Dict[Tuple[str, str], float] = {}

    for tag in tag_list:
        tag_total = tag_counts[tag]

        if tag_total == 0:
            continue

        for word, count in tag_word_counts[tag].items():
            emissions[(word, tag)] = math.log(
                count / tag_total
            )

    return {
        "tags": tag_list,
        "transitions": transitions,
        "unseen_transition_log_prob": unseen_transition_log_prob,
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
    beam_width: int = 40,
) -> List[Tuple[str, str]]:
    """Tag a sentence using trigram-HMM Viterbi decoding.

    The model uses:

        P(tag_i | tag_i-2, tag_i-1)

    Beam pruning is used to keep the computation practical for
    corpora with very large POS tagsets such as the Brown corpus.
    """

    if not words:
        return []

    tags: List[str] = tagger_model["tags"]

    transitions: Dict[Tuple[str, str, str], float] = (
        tagger_model["transitions"]
    )

    unseen_transition_log_prob: float = tagger_model.get(
        "unseen_transition_log_prob",
        -25.0,
    )

    start_transitions: Dict[str, float] = (
        tagger_model["start_transitions"]
    )

    end_transitions: Dict[str, float] = (
        tagger_model["end_transitions"]
    )

    emissions: Dict[Tuple[str, str], float] = (
        tagger_model["emissions"]
    )

    morph_model: MorphologyModel = (
        tagger_model["morphology_model"]
    )

    word_vocab: Set[str] = (
        tagger_model["word_vocab"]
    )

    n = len(words)

    # ---------------------------------------------------------
    # Emission probability helper
    # ---------------------------------------------------------

    def get_emission_log_prob(
        word: str,
        tag: str,
    ) -> float:

        word_lower = word.lower()

        # Known word
        if word_lower in word_vocab:

            pair = (word_lower, tag)

            if pair in emissions:
                return emissions[pair]

            # Seen word but never with this tag.
            return -25.0

        # Unknown word: use morphology.
        return morph_model.log_prob(word, tag)

    # ---------------------------------------------------------
    # One-word sentence
    # ---------------------------------------------------------

    if n == 1:

        best_tag = max(
            tags,
            key=lambda tag:
                start_transitions.get(tag, -20.0)
                + get_emission_log_prob(
                    words[0],
                    tag,
                )
                + end_transitions.get(
                    tag,
                    -20.0,
                ),
        )

        return [(words[0], best_tag)]

    # ---------------------------------------------------------
    # Viterbi state
    #
    # State:
    #
    #     (previous_tag, current_tag)
    #
    # This stores the previous TWO tags.
    # ---------------------------------------------------------

    viterbi: List[
        Dict[Tuple[str, str], float]
    ] = [{} for _ in range(n)]

    backpointer: List[
        Dict[Tuple[str, str], Tuple[str, str]]
    ] = [{} for _ in range(n)]

    # ---------------------------------------------------------
    # First word
    # ---------------------------------------------------------

    for tag in tags:

        score = (
            start_transitions.get(
                tag,
                -20.0,
            )
            + get_emission_log_prob(
                words[0],
                tag,
            )
        )

        viterbi[0][("<s>", tag)] = score

    # ---------------------------------------------------------
    # Second word
    # ---------------------------------------------------------

    second_candidates: List[
        Tuple[float, Tuple[str, str], Tuple[str, str]]
    ] = []

    for previous_tag in tags:

        previous_score = viterbi[0][
            ("<s>", previous_tag)
        ]

        for current_tag in tags:

            transition = transitions.get(
                (
                    "<s>",
                    previous_tag,
                    current_tag,
                ),
                unseen_transition_log_prob,
            )

            score = (
                previous_score
                + transition
                + get_emission_log_prob(
                    words[1],
                    current_tag,
                )
            )

            state = (
                previous_tag,
                current_tag,
            )

            second_candidates.append(
                (
                    score,
                    state,
                    (
                        "<s>",
                        previous_tag,
                    ),
                )
            )

    # Keep only best states.
    second_candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    for score, state, previous_state in second_candidates[
        :beam_width
    ]:
        viterbi[1][state] = score
        backpointer[1][state] = previous_state

    # ---------------------------------------------------------
    # Remaining words
    # ---------------------------------------------------------

    for i in range(2, n):

        word = words[i]

        candidates: List[
            Tuple[float, Tuple[str, str], Tuple[str, str]]
        ] = []

        for (
            prev_prev_tag,
            prev_tag,
        ), previous_score in viterbi[i - 1].items():

            for current_tag in tags:

                transition = transitions.get(
                    (
                        prev_prev_tag,
                        prev_tag,
                        current_tag,
                    ),
                    unseen_transition_log_prob,
                )

                emission = get_emission_log_prob(
                    word,
                    current_tag,
                )

                score = (
                    previous_score
                    + transition
                    + emission
                )

                new_state = (
                    prev_tag,
                    current_tag,
                )

                previous_state = (
                    prev_prev_tag,
                    prev_tag,
                )

                candidates.append(
                    (
                        score,
                        new_state,
                        previous_state,
                    )
                )

        # -----------------------------------------------------
        # Multiple previous states can lead to the same
        # new state. Keep only the best one.
        # -----------------------------------------------------

        best_states: Dict[
            Tuple[str, str],
            Tuple[
                float,
                Tuple[str, str],
            ],
        ] = {}

        for score, state, previous_state in candidates:

            old = best_states.get(state)

            if old is None or score > old[0]:
                best_states[state] = (
                    score,
                    previous_state,
                )

        # -----------------------------------------------------
        # Beam pruning
        # -----------------------------------------------------

        ranked_states = sorted(
            best_states.items(),
            key=lambda x: x[1][0],
            reverse=True,
        )

        for state, (
            score,
            previous_state,
        ) in ranked_states[:beam_width]:

            viterbi[i][state] = score
            backpointer[i][state] = previous_state

    # ---------------------------------------------------------
    # Termination
    # ---------------------------------------------------------

    best_final_score = -float("inf")
    best_final_state = None

    for state, score in viterbi[n - 1].items():

        last_tag = state[1]

        total_score = (
            score
            + end_transitions.get(
                last_tag,
                -20.0,
            )
        )

        if total_score > best_final_score:

            best_final_score = total_score
            best_final_state = state

    if best_final_state is None:
        return [
            (word, tags[0])
            for word in words
        ]

    # ---------------------------------------------------------
    # Traceback
    # ---------------------------------------------------------

    predicted_tags: List[str] = [""] * n

    state = best_final_state

    predicted_tags[n - 1] = state[1]
    predicted_tags[n - 2] = state[0]

    for i in range(n - 1, 1, -1):

        state = backpointer[i][state]

        predicted_tags[i - 2] = state[0]

    return list(zip(words, predicted_tags))


def most_frequent_tag_baseline(
    words: List[str],
    training_data: List[List[Tuple[str, str]]],
) -> List[Tuple[str, str]]:
    """Baseline: assign the most frequent tag for each word."""

    if not words:
        return []

    word_tag_counts: Dict[str, Counter] = defaultdict(Counter)
    global_tag_counts: Counter = Counter()

    for sentence in training_data:

        for word, tag in sentence:

            word_tag_counts[
                word.lower()
            ][tag] += 1

            global_tag_counts[tag] += 1

    default_tag = (
        global_tag_counts.most_common(1)[0][0]
        if global_tag_counts
        else "NOUN"
    )

    word_to_mft: Dict[str, str] = {}

    for word, counts in word_tag_counts.items():

        word_to_mft[word] = (
            counts.most_common(1)[0][0]
        )

    result: List[Tuple[str, str]] = []

    for word in words:

        tag = word_to_mft.get(
            word.lower(),
            default_tag,
        )

        result.append(
            (word, tag)
        )

    return result
