"""Part 5: Live Interactive Continuous Terminal CLI for Spelling Correction.

Features:
- Continuous while-loop accepting user input sentences.
- Full pipeline: Non-word error correction followed by context-aware Real-word error correction.
- ANSI green highlighting + **ASTERISKS** for modified tokens.
- Latency reporting in milliseconds.
- Automatic exit on typing 'exit'.
- Built-in '--test' flag for automated verification on assignment prompt sentences.
"""

import os
import sys
import time
from typing import List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.q3_spelling_corrector.candidate_gen import preprocess_symmetric_delete
from src.q3_spelling_corrector.spell_check import (
    BigramLM,
    build_vocabulary_and_unigram,
    build_bigram_model,
    correct_non_word,
    correct_real_word,
)


def load_models() -> Tuple[set, dict, dict, BigramLM]:
    """Loads Brown corpus and initializes vocabulary, unigram, sym_del dictionary, and bigram model."""
    import nltk
    try:
        from nltk.corpus import brown
        sents = brown.sents()[:35000]
    except (LookupError, AttributeError):
        nltk.download("brown", quiet=True)
        from nltk.corpus import brown
        sents = brown.sents()[:35000]

    vocab, unigram_probs, _ = build_vocabulary_and_unigram(sents)
    sym_del_dict = preprocess_symmetric_delete(vocab)
    bigram_model = build_bigram_model(sents)
    return vocab, unigram_probs, sym_del_dict, bigram_model


def process_sentence(
    sentence: str,
    vocab: set,
    unigram_probs: dict,
    sym_del_dict: dict,
    bigram_model: BigramLM,
) -> Tuple[str, List[Tuple[str, str]], float]:
    """Processes a sentence through non-word and real-word spelling correction.

    Returns:
        Tuple of (formatted_output, list_of_changes, latency_ms).
    """
    t0 = time.perf_counter()

    tokens = sentence.strip().split()
    if not tokens:
        return "", [], 0.0

    # Step 1: Non-word error correction
    non_word_fixed: List[str] = []
    for token in tokens:
        # Separate trailing punctuation
        core = "".join(c for c in token if c.isalnum())
        trailing = "".join(c for c in token if not c.isalnum())
        if core and core.lower() not in vocab:
            corrected_core = correct_non_word(core, vocab, unigram_probs, sym_del_dict, method="B")
            non_word_fixed.append(corrected_core + trailing)
        else:
            non_word_fixed.append(token)

    # Step 2: Real-word error correction
    final_tokens = correct_real_word(non_word_fixed, vocab, bigram_model, sym_del_dict, threshold=2.5)

    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Format output with ANSI colors and asterisks
    changes: List[Tuple[str, str]] = []
    output_tokens: List[str] = []

    for orig, fixed in zip(tokens, final_tokens):
        if orig != fixed:
            changes.append((orig, fixed))
            # ANSI Green + Bold + **Asterisks**
            highlighted = f"\033[1;92m**{fixed}**\033[0m"
            output_tokens.append(highlighted)
        else:
            output_tokens.append(fixed)

    formatted_output = " ".join(output_tokens)
    return formatted_output, changes, latency_ms


def run_cli():
    """Runs the live continuous interactive terminal CLI."""
    print("=" * 65)
    print(" LIVE SPELLING CORRECTOR CLI (Q3 PART 5)")
    print("=" * 65)
    print("Loading models from Brown corpus...")
    vocab, unigram_probs, sym_del_dict, bigram_model = load_models()
    print(f"Model ready! Vocabulary: {len(vocab):,} words.")
    print("Type a sentence to correct, or type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("\033[1;34mInput  >\033[0m ")
            if not user_input.strip():
                continue
            if user_input.strip().lower() == "exit":
                print("Exiting Spelling Corrector CLI. Goodbye!")
                break

            output_text, changes, latency = process_sentence(
                user_input, vocab, unigram_probs, sym_del_dict, bigram_model
            )
            print(f"\033[1;33mOutput >\033[0m {output_text} \033[90m(latency: {latency:.2f} ms)\033[0m")
            if changes:
                change_str = ", ".join(f"{orig} -> {fixed}" for orig, fixed in changes)
                print(f"         \033[90m[Corrections: {change_str}]\033[0m")
            print()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting CLI.")
            break


def run_demo_tests():
    """Runs automated verification on the 4 assignment example sentences."""
    print("=" * 65)
    print(" RUNNING CLI DEMO TESTS ON REQUIRED ASSIGNMENT SENTENCES")
    print("=" * 65)
    vocab, unigram_probs, sym_del_dict, bigram_model = load_models()

    test_sentences = [
        "I hav a good feeling about this.",
        "This is a test sentnce.",
        "I would like to sea the world.",
        "Please meat me at the station.",
    ]

    for sent in test_sentences:
        output_text, changes, latency = process_sentence(
            sent, vocab, unigram_probs, sym_del_dict, bigram_model
        )
        print(f"\nOriginal:  {sent}")
        print(f"Corrected: {output_text} (latency: {latency:.2f} ms)")
        if changes:
            print(f"Changes:   {', '.join(f'{o} -> {c}' for o, c in changes)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_demo_tests()
    else:
        run_cli()
