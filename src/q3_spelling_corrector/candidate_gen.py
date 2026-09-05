"""Candidate Generation Methods for Spelling Correction:
Method A (Standard Edit Distance 1) and Method B (Symmetric Delete).
"""

import string
from collections import defaultdict
from typing import Dict, List, Optional, Set

ALPHABET = string.ascii_lowercase


def damerau_levenshtein_distance_1(w1: str, w2: str) -> bool:
    """Fast check if two words have a Damerau-Levenshtein edit distance <= 1."""
    if w1 == w2:
        return True

    len1, len2 = len(w1), len(w2)
    if abs(len1 - len2) > 1:
        return False

    # Deletion / Insertion
    if len1 > len2:
        # Check if w2 is formed by deleting 1 char from w1
        for i in range(len1):
            if w1[:i] + w1[i + 1 :] == w2:
                return True
        return False
    elif len2 > len1:
        # Check if w1 is formed by deleting 1 char from w2
        for i in range(len2):
            if w2[:i] + w2[i + 1 :] == w1:
                return True
        return False
    else:
        # Same length: Substitution or Transposition
        diffs = [i for i in range(len1) if w1[i] != w2[i]]
        if len(diffs) == 1:
            return True
        elif len(diffs) == 2:
            i, j = diffs
            if j == i + 1 and w1[i] == w2[j] and w1[j] == w2[i]:
                return True
        return False


def generate_candidates_edit1(word: str) -> Set[str]:
    """Method A: Generates all possible strings at an edit distance of 1.

    Generates all deletions, transpositions, replacements, and insertions.
    """
    word_lower = word.lower()
    splits = [(word_lower[:i], word_lower[i:]) for i in range(len(word_lower) + 1)]

    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in ALPHABET]
    inserts = [L + c + R for L, R in splits for c in ALPHABET]

    return set(deletes + transposes + replaces + inserts)


def preprocess_symmetric_delete(vocabulary: Set[str]) -> Dict[str, List[str]]:
    """Method B Preprocessing: Maps every one-character deletion to original vocabulary words.

    Also indexes each word itself so that insertions into misspelled words are
    found in a single lookup.
    """
    preprocessed_dict: Dict[str, List[str]] = defaultdict(list)

    for word in vocabulary:
        w_lower = word.lower()
        # Direct self entry
        preprocessed_dict[w_lower].append(w_lower)

        # 1-character deletions of vocabulary word
        for i in range(len(w_lower)):
            del_variant = w_lower[:i] + w_lower[i + 1 :]
            preprocessed_dict[del_variant].append(w_lower)

    return dict(preprocessed_dict)


def generate_candidates_sym_del(
    word: str,
    preprocessed_dict: Dict[str, List[str]],
    vocabulary: Optional[Set[str]] = None,
) -> Set[str]:
    """Method B Generation: Uses the preprocessed symmetric delete dictionary to find candidates.

    Only deletes from the input word and matches against preprocessed vocabulary deletions.
    """
    word_lower = word.lower()
    candidates: Set[str] = set()

    # 1. Check if word matches directly or is an insertion
    if word_lower in preprocessed_dict:
        for match in preprocessed_dict[word_lower]:
            candidates.add(match)

    # 2. Deletions of input word
    for i in range(len(word_lower)):
        del_variant = word_lower[:i] + word_lower[i + 1 :]
        if del_variant in preprocessed_dict:
            for match in preprocessed_dict[del_variant]:
                candidates.add(match)

    # 3. Transpositions (adjacent character swaps)
    for i in range(len(word_lower) - 1):
        trans = word_lower[:i] + word_lower[i + 1] + word_lower[i] + word_lower[i + 2 :]
        if vocabulary is not None and trans in vocabulary:
            candidates.add(trans)
        elif vocabulary is None and trans in preprocessed_dict:
            candidates.add(trans)

    # Filter to ensure strict edit distance <= 1
    valid_candidates = {c for c in candidates if damerau_levenshtein_distance_1(word_lower, c)}
    return valid_candidates
