"""Evaluation script for Question 3: Spelling Corrector and Speed Demon Benchmark.

Benchmarks:
1. Accuracy on 10% Brown corpus test split:
   - Non-word error correction accuracy.
   - Real-word error correction accuracy.
2. "Speed Demon" Benchmark:
   - Method A (Edit-1) vs. Method B (Symmetric Delete) on exactly 1,000 misspelled words.
   - Latency comparison and algorithmic analysis.
3. Verification on required assignment prompt sentences.
"""

import os
import random
import sys
import time
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.q3_spelling_corrector.candidate_gen import (
    generate_candidates_edit1,
    generate_candidates_sym_del,
    preprocess_symmetric_delete,
)
from src.q3_spelling_corrector.cli import process_sentence
from src.q3_spelling_corrector.spell_check import (
    BigramLM,
    build_bigram_model,
    build_vocabulary_and_unigram,
    correct_non_word,
    correct_real_word,
)


def load_corpus(max_sents: int = 25000) -> Tuple[List[List[str]], List[List[str]]]:
    """Loads Brown corpus and returns 90/10 train/test split."""
    import nltk
    try:
        from nltk.corpus import brown
        sents = brown.sents()[:max_sents]
    except (LookupError, AttributeError):
        nltk.download("brown", quiet=True)
        from nltk.corpus import brown
        sents = brown.sents()[:max_sents]

    cleaned_sents = [[w.strip() for w in s if w.strip()] for s in sents if len(s) >= 3]

    split_idx = int(0.90 * len(cleaned_sents))
    train_sents = cleaned_sents[:split_idx]
    test_sents = cleaned_sents[split_idx:]
    return train_sents, test_sents


def introduce_non_word_error(word: str, vocab: Set[str]) -> Tuple[str, bool]:
    """Introduces a single-edit error creating an out-of-vocabulary word."""
    w = word.lower()
    if len(w) < 3:
        return word, False

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    # Try deletion
    for i in range(len(w)):
        cand = w[:i] + w[i + 1 :]
        if cand not in vocab and len(cand) >= 2:
            return cand, True

    # Try replacement
    for i in range(len(w)):
        for c in alphabet:
            if c != w[i]:
                cand = w[:i] + c + w[i + 1 :]
                if cand not in vocab:
                    return cand, True

    return word, False


def introduce_real_word_error(word: str, vocab: Set[str], sym_del_dict: Dict[str, List[str]]) -> Tuple[str, bool]:
    """Introduces a single-edit error creating another valid vocabulary word."""
    w = word.lower()
    if len(w) < 3 or w not in vocab:
        return word, False

    candidates = generate_candidates_sym_del(w, sym_del_dict, vocab)
    valid_alts = [c for c in candidates if c != w and c in vocab]
    if valid_alts:
        return random.choice(valid_alts), True
    return word, False


def run_speed_demon_benchmark(
    vocab: Set[str],
    sym_del_dict: Dict[str, List[str]],
    unigram_probs: Dict[str, float],
    batch_size: int = 1000,
) -> Dict[str, float]:
    """Runs the Speed Demon benchmark: exactly 1,000 misspelled words through Method A vs Method B."""
    print("\n" + "=" * 75)
    print(f" SPEED DEMON BENCHMARK (EXACTLY {batch_size:,} MISSPELLED WORDS)")
    print("=" * 75)

    # Generate batch of 1,000 misspelled words
    random.seed(42)
    vocab_list = [w for w in vocab if len(w) >= 4]
    sample_words = random.sample(vocab_list, min(len(vocab_list), batch_size * 2))

    misspelled_batch: List[str] = []
    for w in sample_words:
        corrupted, success = introduce_non_word_error(w, vocab)
        if success:
            misspelled_batch.append(corrupted)
            if len(misspelled_batch) == batch_size:
                break

    print(f"Generated test batch of {len(misspelled_batch)} corrupted non-words.")

    # 1. Benchmark Method A: Standard Edit Distance 1
    t0 = time.perf_counter()
    corrections_a: List[str] = []
    for word in misspelled_batch:
        corr = correct_non_word(word, vocab, unigram_probs, sym_del_dict, method="A")
        corrections_a.append(corr)
    time_a = time.perf_counter() - t0

    # 2. Benchmark Method B: Symmetric Delete (SymSpell)
    t0 = time.perf_counter()
    corrections_b: List[str] = []
    for word in misspelled_batch:
        corr = correct_non_word(word, vocab, unigram_probs, sym_del_dict, method="B")
        corrections_b.append(corr)
    time_b = time.perf_counter() - t0

    speedup = (time_a / time_b) if time_b > 0 else float("inf")
    ms_a = (time_a / batch_size) * 1000.0
    ms_b = (time_b / batch_size) * 1000.0

    print("-" * 75)
    print(f"{'Method':<35} | {'Total Time (s)':<16} | {'Avg Latency (ms/word)':<22}")
    print("-" * 75)
    print(f"{'Method A (Standard Edit Distance 1)':<35} | {time_a:<16.4f} | {ms_a:<22.4f}")
    print(f"{'Method B (Symmetric Delete / SymSpell)':<35} | {time_b:<16.4f} | {ms_b:<22.4f}")
    print("-" * 75)
    print(f"Result: Method B is \033[1;92m{speedup:.1f}x FASTER\033[0m than Method A!")

    return {
        "time_a": time_a,
        "time_b": time_b,
        "ms_a": ms_a,
        "ms_b": ms_b,
        "speedup": speedup,
    }


def evaluate_test_set_accuracy(
    test_sents: List[List[str]],
    vocab: Set[str],
    unigram_probs: Dict[str, float],
    sym_del_dict: Dict[str, List[str]],
    bigram_model: BigramLM,
    num_eval: int = 500,
) -> Dict[str, float]:
    """Evaluates non-word and real-word correction accuracy on 10% test split."""
    print("\n" + "=" * 75)
    print(f" TEST SET ACCURACY EVALUATION ({num_eval} SENTENCES)")
    print("=" * 75)

    random.seed(123)
    sample_sents = test_sents[:num_eval]

    # 1. Non-Word Test Set
    non_word_total = 0
    non_word_correct = 0

    for sent in sample_sents:
        # Pick one valid word to corrupt
        candidates_idx = [i for i, w in enumerate(sent) if len(w) >= 3 and w.lower() in vocab]
        if not candidates_idx:
            continue

        target_idx = random.choice(candidates_idx)
        gold_word = sent[target_idx]
        corrupted, ok = introduce_non_word_error(gold_word, vocab)
        if not ok:
            continue

        pred = correct_non_word(corrupted, vocab, unigram_probs, sym_del_dict, method="B")
        non_word_total += 1
        if pred.lower() == gold_word.lower():
            non_word_correct += 1

    non_word_acc = (non_word_correct / non_word_total * 100.0) if non_word_total else 0.0

    # 2. Real-Word Test Set
    real_word_total = 0
    real_word_correct = 0

    for sent in sample_sents:
        # Find words that have edit-1 vocabulary alternatives
        alts_map = {}
        for i, w in enumerate(sent):
            w_lower = w.lower()
            if len(w_lower) >= 3 and w_lower in vocab:
                cands = [c for c in generate_candidates_sym_del(w_lower, sym_del_dict, vocab) if c != w_lower]
                if cands:
                    alts_map[i] = cands

        if not alts_map:
            continue

        target_idx = random.choice(list(alts_map.keys()))
        gold_word = sent[target_idx]
        corrupted_word = random.choice(alts_map[target_idx])

        corrupted_sent = list(sent)
        corrupted_sent[target_idx] = corrupted_word

        fixed_sent = correct_real_word(corrupted_sent, vocab, bigram_model, sym_del_dict, threshold=1.0)
        real_word_total += 1
        if fixed_sent[target_idx].lower() == gold_word.lower():
            real_word_correct += 1

    real_word_acc = (real_word_correct / real_word_total * 100.0) if real_word_total else 0.0

    print(f"Non-Word Error Correction Accuracy: {non_word_acc:.2f}% ({non_word_correct}/{non_word_total})")
    print(f"Real-Word Error Correction Accuracy: {real_word_acc:.2f}% ({real_word_correct}/{real_word_total})")

    return {
        "non_word_acc": non_word_acc,
        "real_word_acc": real_word_acc,
        "non_word_total": non_word_total,
        "real_word_total": real_word_total,
    }


def run_assignment_sample_sentences(vocab, unigram_probs, sym_del_dict, bigram_model):
    """Executes the 4 required example sentences from page 8 of the assignment PDF."""
    print("\n" + "=" * 75)
    print(" SAMPLE SENTENCES (FROM ASSIGNMENT PDF PAGE 8)")
    print("=" * 75)

    test_sentences = [
        ("Non-Word Error", "I hav a good feeling about this."),
        ("Non-Word Error", "This is a test sentnce."),
        ("Real-Word Error", "I would like to sea the world."),
        ("Real-Word Error", "Please meat me at the station."),
    ]

    for category, sentence in test_sentences:
        formatted, changes, latency = process_sentence(
            sentence, vocab, unigram_probs, sym_del_dict, bigram_model
        )
        print(f"\n[{category}]")
        print(f"Original:  {sentence}")
        print(f"Corrected: {formatted} (Latency: {latency:.2f} ms)")
        if changes:
            print(f"Changes:   {', '.join(f'{o} -> {c}' for o, c in changes)}")


def main():
    print("=" * 75)
    print(" QUESTION 3: SPELLING CORRECTOR FULL BENCHMARK")
    print("=" * 75)

    # 1. Load data & models
    print("\n>>> 1. Loading Brown Corpus (90/10 split)...")
    train_sents, test_sents = load_corpus(max_sents=25000)
    print(f"Train sentences: {len(train_sents):,} | Test sentences: {len(test_sents):,}")

    t0 = time.time()
    vocab, unigram_probs, counts = build_vocabulary_and_unigram(train_sents)
    sym_del_dict = preprocess_symmetric_delete(vocab)
    bigram_model = build_bigram_model(train_sents)
    print(f"Models prepared in {time.time() - t0:.2f}s. Vocabulary size: {len(vocab):,}")

    # 2. Accuracy evaluation
    acc_results = evaluate_test_set_accuracy(
        test_sents, vocab, unigram_probs, sym_del_dict, bigram_model, num_eval=500
    )

    # 3. Speed Demon Benchmark
    speed_results = run_speed_demon_benchmark(
        vocab, sym_del_dict, unigram_probs, batch_size=1000
    )

    # 4. Sample sentences verification
    run_assignment_sample_sentences(vocab, unigram_probs, sym_del_dict, bigram_model)

    return {
        "accuracy": acc_results,
        "speed": speed_results,
    }


if __name__ == "__main__":
    main()
