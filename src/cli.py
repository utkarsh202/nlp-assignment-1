#!/usr/bin/env python3
"""
Interactive Terminal CLI for Spelling Correction (Question 3 - Part 5)

Features:
- Continuously prompts the user for sentences.
- Corrects non-word errors and real-word contextual errors.
- Highlights changed words with asterisks and optional ANSI colors.
- Measures and displays end-to-end correction latency in milliseconds.
- Gracefully terminates when the user enters 'exit' or 'quit'.
"""

import time
import re
import nltk
from collections import Counter, defaultdict

from q3_funcs import (
    build_delete_index,
    method_a_candidates,
    method_b_candidates,
    best_candidate_with_context,
    correct_real_word,
)

# ANSI terminal colors for enhanced terminal visualization
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_GREEN = "\033[92m"
COLOR_CYAN = "\033[96m"
COLOR_YELLOW = "\033[93m"


def load_resources():
    """Download Brown corpus and build vocabulary, unigram and bigram counts."""
    print(f"{COLOR_CYAN}Loading Brown corpus and building language models...{COLOR_RESET}")
    nltk.download("brown", quiet=True)

    words = [w.lower() for w in nltk.corpus.brown.words() if w.isalpha()]
    uni_counts = Counter(words)
    vocab = set(uni_counts.keys())

    # Build sentence-level bigram transitions
    bi_by_prev = defaultdict(Counter)
    sents = nltk.corpus.brown.sents()
    for sent in sents:
        w_clean = ["<s>"] + [w.lower() for w in sent if w.isalpha()] + ["</s>"]
        for i in range(len(w_clean) - 1):
            bi_by_prev[w_clean[i]][w_clean[i + 1]] += 1

    delete_index = build_delete_index(vocab)
    print(f"{COLOR_GREEN}Model ready! Vocabulary: {len(vocab):,} words. Delete index entries: {len(delete_index):,}{COLOR_RESET}\n")
    return vocab, uni_counts, bi_by_prev, delete_index


def format_correction(original_words, corrected_words, use_ansi=True):
    """
    Format the corrected sentence, wrapping changed words in **asterisks**
    as required by the assignment (Part 5 highlighting requirement).
    Optionally applies ANSI bold green highlight for terminal display.
    """
    output = []
    for original, corrected in zip(original_words, corrected_words):
        if original != corrected:
            if use_ansi:
                output.append(f"{COLOR_BOLD}{COLOR_GREEN}**{corrected}**{COLOR_RESET}")
            else:
                output.append(f"**{corrected}**")
        else:
            output.append(corrected)
    return " ".join(output)


def correct_sentence_cli(sentence, vocab, uni_counts, bi_by_prev, delete_index, method="B"):
    """
    Corrects a sentence considering both non-word and real-word errors.
    Returns:
        corrected_words: List of strings
        changes: List of tuples (original_word, corrected_word)
    """
    # Split tokens while preserving basic punctuation
    raw_tokens = sentence.strip().split()
    if not raw_tokens:
        return [], []

    words = []
    puncts = []
    for tok in raw_tokens:
        clean = "".join(c for c in tok if c.isalnum()).lower()
        punct = "".join(c for c in tok if not c.isalnum())
        words.append(clean)
        puncts.append(punct)

    corrected_words = []
    changes = []
    n = len(words)

    STOP_WORDS = {
        "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
        "which", "this", "that", "these", "those", "then", "just", "so", "than",
        "such", "both", "through", "about", "for", "is", "of", "while", "during",
        "to", "from", "in", "out", "on", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how", "all",
        "any", "both", "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "can", "will", "just", "don", "should", "now", "i", "me", "my", "myself",
        "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
        "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
        "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "having", "do", "does", "did", "doing", "would", "should", "could",
        "ought"
    }

    for i, word in enumerate(words):
        if not word:
            corrected_words.append(puncts[i])
            continue

        prev_word = corrected_words[i - 1].lower() if i > 0 and corrected_words[i - 1].isalpha() else "<s>"
        next_word = words[i + 1].lower() if i < n - 1 and words[i + 1].isalpha() else "</s>"

        if word not in vocab:
            # 1. Non-word error correction
            if method.upper() == "A":
                cands = method_a_candidates(word, vocab)
            else:
                cands = method_b_candidates(word, delete_index)

            correction = best_candidate_with_context(word, cands, uni_counts, prev_word=prev_word, bi_by_prev=bi_by_prev)
            if not correction:
                correction = word
        else:
            # 2. Real-word contextual error correction
            if word not in STOP_WORDS and len(word) >= 3:
                correction = correct_real_word(
                    word,
                    prev_word,
                    next_word,
                    delete_index,
                    bi_by_prev,
                    uni_counts,
                    len(vocab),
                    improvement_threshold=3.0,
                    method=method,
                    vocab=vocab
                )
            else:
                correction = word

        corrected_token = correction + puncts[i]
        corrected_words.append(corrected_token)
        orig_token = word + puncts[i]
        if corrected_token.lower() != orig_token.lower():
            changes.append((orig_token, corrected_token))

    return corrected_words, changes


def interactive_cli():
    """
    Continuous Terminal CLI — required by Question 3 (Part 5).
    """
    print(f"{COLOR_BOLD}======================================================{COLOR_RESET}")
    print(f"{COLOR_BOLD}   NLP Assignment 3 — Interactive Spelling Corrector  {COLOR_RESET}")
    print(f"{COLOR_BOLD}======================================================{COLOR_RESET}")
    print("Commands:")
    print("  Type any sentence to correct non-word and real-word errors.")
    print("  Type 'method A' or 'method B' to toggle candidate generator.")
    print("  Type 'exit' or 'quit' to stop.\n")

    vocab, uni_counts, bi_by_prev, delete_index = load_resources()
    current_method = "B"

    while True:
        try:
            sentence = input(f"{COLOR_YELLOW}Enter sentence:{COLOR_RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not sentence:
            continue

        if sentence.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if sentence.lower() in {"method a", "method b"}:
            current_method = sentence.split()[1].upper()
            print(f"Switched candidate generation to Method {current_method}.\n")
            continue

        original_words = sentence.split()

        start = time.perf_counter()
        corrected_words, changes = correct_sentence_cli(
            sentence, vocab, uni_counts, bi_by_prev, delete_index, method=current_method
        )
        latency = (time.perf_counter() - start) * 1000

        highlighted = format_correction(original_words, corrected_words, use_ansi=True)

        print(f"{COLOR_BOLD}Corrected:{COLOR_RESET} {highlighted}")
        print(f"{COLOR_CYAN}Latency:   {latency:.3f} ms (Method {current_method}){COLOR_RESET}")
        if changes:
            change_str = ", ".join(f"'{o}' -> '{c}'" for o, c in changes)
            print(f"Changes:   {change_str}")
        else:
            print("Changes:   None (Sentence is clean)")
        print()


if __name__ == "__main__":
    interactive_cli()
