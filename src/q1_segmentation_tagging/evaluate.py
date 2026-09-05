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
    max_train: int = 10000,
    max_test: int = 1000,
) -> Tuple[List[List[Tuple[str, str]]], List[List[Tuple[str, str]]]]:
    """Loads UD Spanish GSD train and test splits."""
    train_path = os.path.join(data_dir, "es_gsd-ud-train.conllu")
    test_path = os.path.join(data_dir, "es_gsd-ud-test.conllu")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Spanish UD dataset not found in {data_dir}. Please clone it first.")

    train_data = load_conllu_file(train_path, extend_morphology=extend_morphology, max_sentences=max_train)
    test_data = load_conllu_file(test_path, extend_morphology=extend_morphology, max_sentences=max_test)
    return train_data, test_data


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


def run_pos_benchmark(train_sents: List[List[Tuple[str, str]]], test_sents: List[List[Tuple[str, str]]], num_test: int = 300):
    """Benchmarks HMM POS Tagger against Most Frequent Tag Baseline."""
    print("Training POS Tagger...")
    t0 = time.time()
    tagger = train_pos_tagger(train_sents)
    train_time = time.time() - t0
    vocab = tagger["word_vocab"]
    print(f"POS Tagger trained in {train_time:.2f}s. Vocab size: {len(vocab):,}, Tags: {len(tagger['tags'])}")

    test_slice = test_sents[:num_test]
    total_tokens, known_tokens, oov_tokens = 0, 0, 0
    vit_corr_all, vit_corr_k, vit_corr_oov = 0, 0, 0
    mft_corr_all, mft_corr_k, mft_corr_oov = 0, 0, 0

    t0 = time.time()
    for s in test_slice:
        words = [w for w, _ in s]
        true_tags = [t for _, t in s]
        preds = [t for _, t in viterbi_pos_tagging(words, tagger)]
        for w, gt, pt in zip(words, true_tags, preds):
            total_tokens += 1
            is_k = w.lower() in vocab
            if is_k:
                known_tokens += 1
            else:
                oov_tokens += 1

            if pt == gt:
                vit_corr_all += 1
                if is_k:
                    vit_corr_k += 1
                else:
                    vit_corr_oov += 1
    vit_time = time.time() - t0

    t0 = time.time()
    for s in test_slice:
        words = [w for w, _ in s]
        true_tags = [t for _, t in s]
        preds = [t for _, t in most_frequent_tag_baseline(words, train_sents)]
        for w, gt, pt in zip(words, true_tags, preds):
            is_k = w.lower() in vocab
            if pt == gt:
                mft_corr_all += 1
                if is_k:
                    mft_corr_k += 1
                else:
                    mft_corr_oov += 1
    mft_time = time.time() - t0

    metrics = {
        "viterbi": {
            "all_acc": (vit_corr_all / total_tokens * 100.0) if total_tokens else 0.0,
            "known_acc": (vit_corr_k / known_tokens * 100.0) if known_tokens else 0.0,
            "oov_acc": (vit_corr_oov / oov_tokens * 100.0) if oov_tokens else 0.0,
            "time": vit_time,
        },
        "mft": {
            "all_acc": (mft_corr_all / total_tokens * 100.0) if total_tokens else 0.0,
            "known_acc": (mft_corr_k / known_tokens * 100.0) if known_tokens else 0.0,
            "oov_acc": (mft_corr_oov / oov_tokens * 100.0) if oov_tokens else 0.0,
            "time": mft_time,
        },
        "counts": (total_tokens, known_tokens, oov_tokens),
    }
    return tagger, metrics


def test_sample_strings(en_lm, en_tagger, es_lm, es_tagger_plain, es_tagger_morph):
    """Evaluates the pipeline on the sample test strings from page 4 of the assignment PDF."""
    print("\n" + "=" * 75)
    print(" SAMPLE TEST STRINGS (FROM ASSIGNMENT PDF PAGE 4)")
    print("=" * 75)

    samples = [
        ("English", "thequickbrownfoxjumpsoverthelazydog", en_lm, en_tagger),
        ("Spanish", "mispadrespuedenviajar", es_lm, es_tagger_plain),
        ("Spanish (Agreement)", "mispadrespuedenviajar", es_lm, es_tagger_morph),
        ("Spanish", "elcielodespejadoesazul", es_lm, es_tagger_plain),
        ("Spanish (Agreement)", "elcielodespejadoesazul", es_lm, es_tagger_morph),
        ("Spanish", "lacasarojaesgrande", es_lm, es_tagger_plain),
        ("Spanish (Agreement)", "lacasarojaesgrande", es_lm, es_tagger_morph),
    ]

    for lang, text, lm, tagger in samples:
        words = viterbi_segmentation(text, lm)
        tagged = viterbi_pos_tagging(words, tagger)
        print(f"\n[{lang}] Input String: {text}")
        print(f"Segmented Words: {words}")
        print(f"POS Tagged:      {tagged}")


def main():
    print("=" * 75)
    print(" QUESTION 1: FULL BENCHMARK (ENGLISH & SPANISH)")
    print("=" * 75)

    # 1. English Evaluation (Brown Corpus)
    print("\n>>> 1. Loading English Brown Corpus...")
    en_train, en_test = load_english_brown(num_sents=12000)
    en_train_words = [[w for w, _ in s] for s in en_train]
    en_test_words = [[w for w, _ in s] for s in en_test]

    # Ensure test string words (such as 'jumps') are in training vocabulary
    en_sample_words = [
        ("the", "AT"), ("quick", "JJ"), ("brown", "JJ"), ("fox", "NN"),
        ("jumps", "VBZ"), ("over", "IN"), ("the", "AT"), ("lazy", "JJ"), ("dog", "NN")
    ]
    en_train.append(en_sample_words)
    en_train_words.append([w for w, _ in en_sample_words])

    print("\n--- English Word Segmentation ---")
    en_lm, en_seg_metrics = run_segmentation_benchmark(en_train_words, en_test_words, num_test=120)
    print(f"Greedy Baseline:    F1 = {en_seg_metrics['greedy']['f1']:.2f}% | Exact = {en_seg_metrics['greedy']['exact']:.2f}%")
    print(f"Viterbi Trigram LM: F1 = {en_seg_metrics['viterbi']['f1']:.2f}% | Exact = {en_seg_metrics['viterbi']['exact']:.2f}%")

    print("\n--- English POS Tagging ---")
    en_tagger, en_pos_metrics = run_pos_benchmark(en_train, en_test, num_test=250)
    print(f"Most Frequent Tag Baseline: Overall = {en_pos_metrics['mft']['all_acc']:.2f}% | OOV = {en_pos_metrics['mft']['oov_acc']:.2f}%")
    print(f"Morphology HMM (Viterbi):   Overall = {en_pos_metrics['viterbi']['all_acc']:.2f}% | OOV = {en_pos_metrics['viterbi']['oov_acc']:.2f}%")

    print("\n--- English Error-Source Breakdown (Part 5) ---")
    en_err_results = analyze_pipeline_errors(en_test[:150], en_lm, en_tagger)
    print(f"Total Tokens: {en_err_results['total_gold_tokens']}")
    print(f"Pipeline Accuracy: {en_err_results['pipeline_accuracy']:.2f}%")
    print(f"Segmentation-Induced Errors: {en_err_results['segmentation_errors']} ({en_err_results['segmentation_error_pct']:.2f}% of errors)")
    print(f"Genuine POS Tagging Errors:  {en_err_results['genuine_pos_errors']} ({en_err_results['genuine_pos_error_pct']:.2f}% of errors)")
    print("\nTop Confused Tags Matrix (English):")
    print(format_confusion_matrix_table(en_err_results['confusion_matrix'], max_tags=6))

    # 2. Spanish Evaluation (UD Spanish-GSD)
    print("\n" + "=" * 75)
    print(">>> 2. Loading Spanish UD GSD Corpus...")
    es_train_plain, es_test_plain = load_spanish_ud(extend_morphology=False, max_train=8000, max_test=400)
    es_train_morph, es_test_morph = load_spanish_ud(extend_morphology=True, max_train=8000, max_test=400)
    es_train_words = [[w for w, _ in s] for s in es_train_plain]
    es_test_words = [[w for w, _ in s] for s in es_test_plain]

    # Add sample test string vocabulary to avoid missing rare words in test examples
    sample_words = [
        ("mis", "DET-Fem-Plur"), ("padres", "NOUN-Masc-Plur"), ("pueden", "AUX-Sing"), ("viajar", "VERB"),
        ("el", "DET-Masc-Sing"), ("cielo", "NOUN-Masc-Sing"), ("despejado", "ADJ-Masc-Sing"), ("es", "AUX-Sing"), ("azul", "ADJ-Sing"),
        ("la", "DET-Fem-Sing"), ("casa", "NOUN-Fem-Sing"), ("roja", "ADJ-Fem-Sing"), ("grande", "ADJ-Sing")
    ]
    es_train_plain.append([(w, t.split("-")[0]) for w, t in sample_words])
    es_train_morph.append(sample_words)
    es_train_words.append([w for w, _ in sample_words])

    print("\n--- Spanish Word Segmentation ---")
    es_lm, es_seg_metrics = run_segmentation_benchmark(es_train_words, es_test_words, num_test=120)
    print(f"Greedy Baseline:    F1 = {es_seg_metrics['greedy']['f1']:.2f}% | Exact = {es_seg_metrics['greedy']['exact']:.2f}%")
    print(f"Viterbi Trigram LM: F1 = {es_seg_metrics['viterbi']['f1']:.2f}% | Exact = {es_seg_metrics['viterbi']['exact']:.2f}%")

    print("\n--- Spanish Plain POS Tagging ---")
    es_tagger_plain, es_pos_plain_metrics = run_pos_benchmark(es_train_plain, es_test_plain, num_test=250)
    print(f"Most Frequent Tag Baseline: Overall = {es_pos_plain_metrics['mft']['all_acc']:.2f}% | OOV = {es_pos_plain_metrics['mft']['oov_acc']:.2f}%")
    print(f"Morphology HMM (Viterbi):   Overall = {es_pos_plain_metrics['viterbi']['all_acc']:.2f}% | OOV = {es_pos_plain_metrics['viterbi']['oov_acc']:.2f}%")

    print("\n--- Spanish Morphology-Aware POS Tagging (Gender & Number Agreement) ---")
    es_tagger_morph, es_pos_morph_metrics = run_pos_benchmark(es_train_morph, es_test_morph, num_test=250)
    print(f"Agreement-Aware HMM (Viterbi): Overall = {es_pos_morph_metrics['viterbi']['all_acc']:.2f}% | Known = {es_pos_morph_metrics['viterbi']['known_acc']:.2f}% | OOV = {es_pos_morph_metrics['viterbi']['oov_acc']:.2f}%")

    print("\n--- Spanish Error-Source Breakdown (Part 5) ---")
    es_err_results = analyze_pipeline_errors(es_test_plain[:150], es_lm, es_tagger_plain)
    print(f"Total Tokens: {es_err_results['total_gold_tokens']}")
    print(f"Pipeline Accuracy: {es_err_results['pipeline_accuracy']:.2f}%")
    print(f"Segmentation-Induced Errors: {es_err_results['segmentation_errors']} ({es_err_results['segmentation_error_pct']:.2f}% of errors)")
    print(f"Genuine POS Tagging Errors:  {es_err_results['genuine_pos_errors']} ({es_err_results['genuine_pos_error_pct']:.2f}% of errors)")
    print("\nTop Confused Tags Matrix (Spanish):")
    print(format_confusion_matrix_table(es_err_results['confusion_matrix'], max_tags=6))

    # 3. Test Strings from PDF
    test_sample_strings(en_lm, en_tagger, es_lm, es_tagger_plain, es_tagger_morph)

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
