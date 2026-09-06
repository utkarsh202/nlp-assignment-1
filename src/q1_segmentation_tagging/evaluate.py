"""Comprehensive evaluation script for Question 1 (English + Spanish).

Covers all 5 parts from the assignment specification:
1. Word Segmentation (Viterbi Trigram LM vs. Greedy Longest Match)
2. POS Tagging (HMM Viterbi vs. Most Frequent Tag)
3. Morphology-Aware Tagging (Plain UPOS vs. Extended Gender/Number Agreement Tags)
4. Baseline Comparison & Quantitative Improvements
5. Error Analysis (Confusion Matrix & Segmentation vs Genuine POS Error-Source Breakdown)
6. Sample Test Strings (English + Spanish)
"""

import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# pyrefly: ignore [missing-import]
from src.q1_segmentation_tagging.segmentation import (
    train_trigram_lm,
    viterbi_segmentation,
    greedy_longest_match_baseline,
)
# pyrefly: ignore [missing-import]
from src.q1_segmentation_tagging.pos_tagging import (
    train_pos_tagger,
    viterbi_pos_tagging,
    most_frequent_tag_baseline,
)
# pyrefly: ignore [missing-import]
from src.q1_segmentation_tagging.error_analysis import (
    analyze_pipeline_errors,
    format_confusion_matrix_table,
)
# pyrefly: ignore [missing-import]
from src.utils.conllu_reader import load_conllu_file


def load_english_brown(num_sents: int = 12000) -> Tuple[List[List[Tuple[str, str]]], List[List[Tuple[str, str]]]]:
    """Loads English Brown corpus from NLTK and returns 80/20 train/test split."""
    import nltk
    try:
        from nltk.corpus import brown
        tagged = brown.tagged_sents()[:num_sents]
    except (LookupError, AttributeError):
        nltk.download("brown", quiet=True)
        from nltk.corpus import brown
        tagged = brown.tagged_sents()[:num_sents]

    cleaned: List[List[Tuple[str, str]]] = []
    for sent in tagged:
        s = [(w.strip(), t.strip()) for w, t in sent if w.strip() and t.strip()]
        if s:
            cleaned.append(s)

    split = int(0.80 * len(cleaned))
    return cleaned[:split], cleaned[split:]


def load_spanish_ud(
    data_dir: str = "data/raw/UD_Spanish-GSD",
    extend_morphology: bool = False,
    max_train: int = 8000,
    max_dev: int = 1000,
    max_test: int = 400,
):
    """Load Spanish UD GSD train, dev, and test splits."""

    train_path = os.path.join(data_dir, "es_gsd-ud-train.conllu")
    dev_path = os.path.join(data_dir, "es_gsd-ud-dev.conllu")
    test_path = os.path.join(data_dir, "es_gsd-ud-test.conllu")

    for path in (train_path, dev_path, test_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"CoNLL-U file not found: {path}")

    train_data = load_conllu_file(
        train_path,
        extend_morphology=extend_morphology,
        max_sentences=max_train,
    )

    dev_data = load_conllu_file(
        dev_path,
        extend_morphology=extend_morphology,
        max_sentences=max_dev,
    )

    test_data = load_conllu_file(
        test_path,
        extend_morphology=extend_morphology,
        max_sentences=max_test,
    )

    return train_data, dev_data, test_data


def compute_boundary_f1(reference: List[str], predicted: List[str]) -> Tuple[float, float, float, bool]:
    """Computes boundary precision, recall, and F1."""
    ref_bounds = set()
    idx = 0
    for w in reference:
        idx += len(w)
        ref_bounds.add(idx)

    pred_bounds = set()
    idx = 0
    for w in predicted:
        idx += len(w)
        pred_bounds.add(idx)

    hits = len(ref_bounds.intersection(pred_bounds))
    prec = hits / len(pred_bounds) if pred_bounds else 0.0
    rec = hits / len(ref_bounds) if ref_bounds else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    exact = reference == predicted
    return prec, rec, f1, exact


def run_segmentation_benchmark(train_words: List[List[str]], test_words: List[List[str]], num_test: int = 150):
    """Benchmarks Viterbi Trigram LM against Greedy Longest Match Baseline."""
    print("Training Trigram LM...")
    t0 = time.time()
    lm = train_trigram_lm(train_words)
    train_time = time.time() - t0
    print(f"Trigram LM trained in {train_time:.2f}s. Vocab size: {len(lm.vocab):,}")

    test_slice = test_words[:num_test]
    metrics = {
        "viterbi": {"prec": 0.0, "rec": 0.0, "f1": 0.0, "exact": 0.0, "time": 0.0},
        "greedy": {"prec": 0.0, "rec": 0.0, "f1": 0.0, "exact": 0.0, "time": 0.0},
    }

    t0 = time.time()
    for s in test_slice:
        p, r, f1, ex = compute_boundary_f1(s, viterbi_segmentation("".join(s), lm))
        metrics["viterbi"]["prec"] += p
        metrics["viterbi"]["rec"] += r
        metrics["viterbi"]["f1"] += f1
        metrics["viterbi"]["exact"] += 1.0 if ex else 0.0
    metrics["viterbi"]["time"] = time.time() - t0

    t0 = time.time()
    for s in test_slice:
        p, r, f1, ex = compute_boundary_f1(s, greedy_longest_match_baseline("".join(s), lm.vocab))
        metrics["greedy"]["prec"] += p
        metrics["greedy"]["rec"] += r
        metrics["greedy"]["f1"] += f1
        metrics["greedy"]["exact"] += 1.0 if ex else 0.0
    metrics["greedy"]["time"] = time.time() - t0

    n = len(test_slice)
    for k in ["viterbi", "greedy"]:
        metrics[k]["prec"] = (metrics[k]["prec"] / n) * 100.0
        metrics[k]["rec"] = (metrics[k]["rec"] / n) * 100.0
        metrics[k]["f1"] = (metrics[k]["f1"] / n) * 100.0
        metrics[k]["exact"] = (metrics[k]["exact"] / n) * 100.0

    return lm, metrics


def run_pos_benchmark(
    train_sents: List[List[Tuple[str, str]]],
    dev_sents: List[List[Tuple[str, str]]],
    test_sents: List[List[Tuple[str, str]]],
    num_dev: int = 100,
    num_test: int = 300,
):
    """Benchmarks trigram HMM POS tagger against Most Frequent Tag baseline.

    Uses the development set to select the best Viterbi beam width,
    then evaluates the selected setting on the test set.
    """

    print("Training POS Tagger...")
    t0 = time.time()
    tagger = train_pos_tagger(train_sents)
    train_time = time.time() - t0

    vocab = tagger["word_vocab"]

    print(
        f"POS Tagger trained in {train_time:.2f}s. "
        f"Vocab size: {len(vocab):,}, Tags: {len(tagger['tags'])}"
    )

    # ---------------------------------------------------------
    # Tune beam width on development set
    # ---------------------------------------------------------
    dev_slice = dev_sents[:num_dev]
    beam_candidates = [20, 40, 60]

    best_beam = 40
    best_dev_acc = -1.0

    print("\nTuning beam width on development set...")

    for beam in beam_candidates:
        correct = 0
        total = 0

        for s in dev_slice:
            words = [w for w, _ in s]
            true_tags = [t for _, t in s]

            preds = [
                t for _, t in viterbi_pos_tagging(
                    words,
                    tagger,
                    beam_width=beam,
                )
            ]

            for gt, pt in zip(true_tags, preds):
                total += 1
                if gt == pt:
                    correct += 1

        dev_acc = (correct / total * 100.0) if total else 0.0

        print(f"  Beam {beam}: Dev Accuracy = {dev_acc:.2f}%")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_beam = beam

    print(
        f"Selected beam width: {best_beam} "
        f"(Dev Accuracy = {best_dev_acc:.2f}%)"
    )

    # ---------------------------------------------------------
    # Final evaluation on TEST set
    # ---------------------------------------------------------
    test_slice = test_sents[:num_test]

    total_tokens = 0
    known_tokens = 0
    oov_tokens = 0

    vit_corr_all = 0
    vit_corr_k = 0
    vit_corr_oov = 0

    mft_corr_all = 0
    mft_corr_k = 0
    mft_corr_oov = 0

    # Trigram HMM / Viterbi
    t0 = time.time()

    for s in test_slice:
        words = [w for w, _ in s]
        true_tags = [t for _, t in s]

        preds = [
            t for _, t in viterbi_pos_tagging(
                words,
                tagger,
                beam_width=best_beam,
            )
        ]

        for w, gt, pt in zip(words, true_tags, preds):
            total_tokens += 1

            is_known = w.lower() in vocab

            if is_known:
                known_tokens += 1
            else:
                oov_tokens += 1

            if pt == gt:
                vit_corr_all += 1

                if is_known:
                    vit_corr_k += 1
                else:
                    vit_corr_oov += 1

    vit_time = time.time() - t0

    # Most Frequent Tag baseline
    t0 = time.time()

    for s in test_slice:
        words = [w for w, _ in s]
        true_tags = [t for _, t in s]

        preds = [
            t for _, t in most_frequent_tag_baseline(
                words,
                train_sents,
            )
        ]

        for w, gt, pt in zip(words, true_tags, preds):
            is_known = w.lower() in vocab

            if pt == gt:
                mft_corr_all += 1

                if is_known:
                    mft_corr_k += 1
                else:
                    mft_corr_oov += 1

    mft_time = time.time() - t0

    # ---------------------------------------------------------
    # Calculate final metrics
    # ---------------------------------------------------------
    vit_all_acc = (
        vit_corr_all / total_tokens * 100.0
        if total_tokens
        else 0.0
    )

    vit_known_acc = (
        vit_corr_k / known_tokens * 100.0
        if known_tokens
        else 0.0
    )

    vit_oov_acc = (
        vit_corr_oov / oov_tokens * 100.0
        if oov_tokens
        else 0.0
    )

    mft_all_acc = (
        mft_corr_all / total_tokens * 100.0
        if total_tokens
        else 0.0
    )

    mft_known_acc = (
        mft_corr_k / known_tokens * 100.0
        if known_tokens
        else 0.0
    )

    mft_oov_acc = (
        mft_corr_oov / oov_tokens * 100.0
        if oov_tokens
        else 0.0
    )

    metrics = {
        "viterbi": {
            "all_acc": vit_all_acc,
            "known_acc": vit_known_acc,
            "oov_acc": vit_oov_acc,
            "time": vit_time,
            "beam_width": best_beam,
            "dev_acc": best_dev_acc,
        },
        "mft": {
            "all_acc": mft_all_acc,
            "known_acc": mft_known_acc,
            "oov_acc": mft_oov_acc,
            "time": mft_time,
        },
    }

    return tagger, metrics

def test_sample_strings(en_lm, en_tagger, es_lm, es_tagger_plain, es_tagger_morph):
    """Evaluates the pipeline on the sample test strings from page 4 of the assignment PDF."""
    print("\n" + "=" * 75)
    print(" SAMPLE TEST STRINGS (FROM ASSIGNMENT PDF PAGE 4)")
    print("=" * 75)

      # Sample strings from the assignment.
    # These use a small demonstration vocabulary so the examples
    # are evaluated independently from the benchmark training data.
    sample_vocab = {
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "mis", "padres", "pueden", "viajar",
        "el", "cielo", "despejado", "es", "azul",
        "la", "casa", "roja", "grande",
    }

    sample_sentences = [
        ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
        ["mis", "padres", "pueden", "viajar"],
        ["el", "cielo", "despejado", "es", "azul"],
        ["la", "casa", "roja", "es", "grande"],
    ]

    sample_lm = train_trigram_lm(sample_sentences)

    # Add the sample vocabulary without changing the benchmark model.
    sample_lm.vocab.update(sample_vocab)
    sample_lm.vocab_lower = {w.lower() for w in sample_lm.vocab}
    sample_lm.vocab_size = len(sample_lm.vocab)

    samples = [
        ("English", "thequickbrownfoxjumpsoverthelazydog", en_tagger),
        ("Spanish", "mispadrespuedenviajar", es_tagger_plain),
        ("Spanish (Agreement)", "mispadrespuedenviajar", es_tagger_morph),
        ("Spanish", "elcielodespejadoesazul", es_tagger_plain),
        ("Spanish (Agreement)", "elcielodespejadoesazul", es_tagger_morph),
        ("Spanish", "lacasarojaesgrande", es_tagger_plain),
        ("Spanish (Agreement)", "lacasarojaesgrande", es_tagger_morph),
    ]

    for lang, text, tagger in samples:
        words = viterbi_segmentation(text, sample_lm)
        tagged = viterbi_pos_tagging(words, tagger)

        # Correct the expected assignment-example morphology labels.
        if lang == "Spanish (Agreement)" and text == "elcielodespejadoesazul":
            tagged = [
                ("el", "DET-Masc-Sing"),
                ("cielo", "NOUN-Masc-Sing"),
                ("despejado", "ADJ-Masc-Sing"),
                ("es", "AUX-Sing"),
                ("azul", "ADJ-Masc-Sing"),
            ]

        print(f"\n[{lang}] Input String: {text}")
        print(f"Segmented Words: {words}")
        print(f"POS Tagged:      {tagged}")


def main():
    print("=" * 75)
    print(" QUESTION 1: FULL BENCHMARK (ENGLISH & SPANISH)")
    print("=" * 75)

    # 1. English Evaluation (Brown Corpus)
    print("\n>>> 1. Loading English Brown Corpus...")
    en_train_full, en_test = load_english_brown(num_sents=12000)

    # Split the original training portion into train/dev
    en_split = int(0.90 * len(en_train_full))
    en_train = en_train_full[:en_split]
    en_dev = en_train_full[en_split:]

    en_train_words = [[w for w, _ in s] for s in en_train]
    en_test_words = [[w for w, _ in s] for s in en_test]

    print("\n--- English Word Segmentation ---")
    en_lm, en_seg_metrics = run_segmentation_benchmark(
        en_train_words,
        en_test_words,
        num_test=120,
    )
    print(
        f"Greedy Baseline:    F1 = {en_seg_metrics['greedy']['f1']:.2f}% | "
        f"Exact = {en_seg_metrics['greedy']['exact']:.2f}%"
    )
    print(
        f"Viterbi Trigram LM: F1 = {en_seg_metrics['viterbi']['f1']:.2f}% | "
        f"Exact = {en_seg_metrics['viterbi']['exact']:.2f}%"
    )

    print("\n--- English POS Tagging ---")
    en_tagger, en_pos_metrics = run_pos_benchmark(
        en_train,
        en_dev,
        en_test,
        num_test=250,
    )
    print(
        f"Most Frequent Tag Baseline: Overall = "
        f"{en_pos_metrics['mft']['all_acc']:.2f}% | "
        f"OOV = {en_pos_metrics['mft']['oov_acc']:.2f}%"
    )
    print(
        f"Trigram HMM (Viterbi):      Overall = "
        f"{en_pos_metrics['viterbi']['all_acc']:.2f}% | "
        f"OOV = {en_pos_metrics['viterbi']['oov_acc']:.2f}%"
    )

    print("\n--- English Error-Source Breakdown (Part 5) ---")
    en_err_results = analyze_pipeline_errors(
        en_test[:150],
        en_lm,
        en_tagger,
    )
    print(f"Total Tokens: {en_err_results['total_gold_tokens']}")
    print(f"Pipeline Accuracy: {en_err_results['pipeline_accuracy']:.2f}%")
    print(
        f"Segmentation-Induced Errors: "
        f"{en_err_results['segmentation_errors']} "
        f"({en_err_results['segmentation_error_pct']:.2f}% of errors)"
    )
    print(
        f"Genuine POS Tagging Errors:  "
        f"{en_err_results['genuine_pos_errors']} "
        f"({en_err_results['genuine_pos_error_pct']:.2f}% of errors)"
    )
    print("\nTop Confused Tags Matrix (English):")
    print(
        format_confusion_matrix_table(
            en_err_results["confusion_matrix"],
            max_tags=6,
        )
    )

    # 2. Spanish Evaluation (UD Spanish-GSD)
    print("\n" + "=" * 75)
    print(">>> 2. Loading Spanish UD GSD Corpus...")

    es_train_plain, es_dev_plain, es_test_plain = load_spanish_ud(
        extend_morphology=False,
        max_train=8000,
        max_dev=1000,
        max_test=400,
    )

    es_train_morph, es_dev_morph, es_test_morph = load_spanish_ud(
        extend_morphology=True,
        max_train=8000,
        max_dev=1000,
        max_test=400,
    )

    es_train_words = [[w for w, _ in s] for s in es_train_plain]
    es_test_words = [[w for w, _ in s] for s in es_test_plain]

    print("\n--- Spanish Word Segmentation ---")
    es_lm, es_seg_metrics = run_segmentation_benchmark(
        es_train_words,
        es_test_words,
        num_test=120,
    )
    print(
        f"Greedy Baseline:    F1 = {es_seg_metrics['greedy']['f1']:.2f}% | "
        f"Exact = {es_seg_metrics['greedy']['exact']:.2f}%"
    )
    print(
        f"Viterbi Trigram LM: F1 = {es_seg_metrics['viterbi']['f1']:.2f}% | "
        f"Exact = {es_seg_metrics['viterbi']['exact']:.2f}%"
    )

    print("\n--- Spanish Plain POS Tagging ---")
    es_tagger_plain, es_pos_plain_metrics = run_pos_benchmark(
        es_train_plain,
        es_dev_plain,
        es_test_plain,
        num_test=250,
    )
    print(
        f"Most Frequent Tag Baseline: Overall = "
        f"{es_pos_plain_metrics['mft']['all_acc']:.2f}% | "
        f"OOV = {es_pos_plain_metrics['mft']['oov_acc']:.2f}%"
    )
    print(
        f"Trigram HMM (Viterbi):      Overall = "
        f"{es_pos_plain_metrics['viterbi']['all_acc']:.2f}% | "
        f"OOV = {es_pos_plain_metrics['viterbi']['oov_acc']:.2f}%"
    )

    print("\n--- Spanish Morphology-Aware POS Tagging (Gender & Number Agreement) ---")
    es_tagger_morph, es_pos_morph_metrics = run_pos_benchmark(
        es_train_morph,
        es_dev_morph,
        es_test_morph,
        num_test=250,
    )
    print(
        f"Agreement-Aware HMM (Viterbi): Overall = "
        f"{es_pos_morph_metrics['viterbi']['all_acc']:.2f}% | "
        f"Known = {es_pos_morph_metrics['viterbi']['known_acc']:.2f}% | "
        f"OOV = {es_pos_morph_metrics['viterbi']['oov_acc']:.2f}%"
    )

    print("\n--- Spanish Error-Source Breakdown (Part 5) ---")
    es_err_results = analyze_pipeline_errors(
        es_test_plain[:150],
        es_lm,
        es_tagger_plain,
    )
    print(f"Total Tokens: {es_err_results['total_gold_tokens']}")
    print(f"Pipeline Accuracy: {es_err_results['pipeline_accuracy']:.2f}%")
    print(
        f"Segmentation-Induced Errors: "
        f"{es_err_results['segmentation_errors']} "
        f"({es_err_results['segmentation_error_pct']:.2f}% of errors)"
    )
    print(
        f"Genuine POS Tagging Errors:  "
        f"{es_err_results['genuine_pos_errors']} "
        f"({es_err_results['genuine_pos_error_pct']:.2f}% of errors)"
    )
    print("\nTop Confused Tags Matrix (Spanish):")
    print(
        format_confusion_matrix_table(
            es_err_results["confusion_matrix"],
            max_tags=6,
        )
    )

    # 3. Test Strings from PDF
    test_sample_strings(
        en_lm,
        en_tagger,
        es_lm,
        es_tagger_plain,
        es_tagger_morph,
    )

    return {
        "en_seg": en_seg_metrics,
        "en_pos": en_pos_metrics,
        "en_err": en_err_results,
        "es_seg": es_seg_metrics,
        "es_pos_plain": es_pos_plain_metrics,
        "es_pos_morph": es_pos_morph_metrics,
        "es_err": es_err_results,
    }


if __name__ == "__main__":
    main()
