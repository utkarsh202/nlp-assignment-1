"""Question 2: Transition-Based Dependency Parser (Arc-Standard)."""

from .parser import (
    Configuration,
    ParsedSentence,
    parse_conllu,
    simulate_oracle,
    parse_sentence,
)
from .feature_extraction import (
    extract_features,
    train_classifier,
)

__all__ = [
    "Configuration",
    "ParsedSentence",
    "parse_conllu",
    "simulate_oracle",
    "parse_sentence",
    "extract_features",
    "train_classifier",
]
