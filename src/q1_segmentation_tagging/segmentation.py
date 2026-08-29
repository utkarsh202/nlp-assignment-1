from typing import List, Tuple, Dict, Any

def train_trigram_lm(corpus: List[List[str]]) -> Any:
    """Trains a trigram language model on the given corpus."""
    pass

def viterbi_segmentation(text: str, lm_model: Any) -> List[str]:
    """Segments a continuous string of text into words using the Viterbi algorithm and trigram model."""
    pass

def greedy_longest_match_baseline(text: str, vocabulary: set) -> List[str]:
    """Baseline for segmentation: always match the longest word from vocabulary at each point."""
    pass
