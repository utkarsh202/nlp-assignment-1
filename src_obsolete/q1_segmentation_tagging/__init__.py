"""Question 1: Word Segmentation and Morphology-Aware POS Tagging."""

from .segmentation import (
    TrigramLM,
    train_trigram_lm,
    viterbi_segmentation,
    greedy_longest_match_baseline,
)
from .pos_tagging import (
    train_pos_tagger,
    viterbi_pos_tagging,
    most_frequent_tag_baseline,
    extract_morphological_features,
    MorphologyModel,
)

__all__ = [
    "TrigramLM",
    "train_trigram_lm",
    "viterbi_segmentation",
    "greedy_longest_match_baseline",
    "train_pos_tagger",
    "viterbi_pos_tagging",
    "most_frequent_tag_baseline",
    "extract_morphological_features",
    "MorphologyModel",
]
