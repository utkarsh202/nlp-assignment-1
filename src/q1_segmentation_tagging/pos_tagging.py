from typing import List, Tuple, Dict, Any

def train_pos_tagger(corpus: List[List[Tuple[str, str]]]) -> Dict[str, Any]:
    """Trains emission and transition probabilities for POS tagging (morphology-aware)."""
    pass

def viterbi_pos_tagging(words: List[str], tagger_model: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Tags a sequence of words using Viterbi decoding and trained emission/transition probabilities."""
    pass

def most_frequent_tag_baseline(words: List[str], training_data: List[List[Tuple[str, str]]]) -> List[Tuple[str, str]]:
    """Baseline for POS tagging: assign the most frequent tag observed in training data."""
    pass
