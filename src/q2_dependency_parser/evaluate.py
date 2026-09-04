"""Evaluation script for Question 2: Transition-Based Dependency Parser.

Trains on UD_English-EWT (en_ewt-ud-train.conllu) and evaluates on dev set (en_ewt-ud-dev.conllu).
Calculates:
- Labeled Attachment Score (LAS)
- Unlabeled Attachment Score (UAS)
- Dependency parse trees on the required sample sentences
"""

import os
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.q2_dependency_parser.parser import (
    ParsedSentence,
    parse_conllu,
    simulate_oracle,
    parse_sentence,
)
from src.q2_dependency_parser.feature_extraction import train_classifier


def evaluate_dev_set(
    dev_sentences: List[ParsedSentence],
    classifier: Any,
    max_eval: int = 500,
) -> Dict[str, float]:
    """Evaluates the parser on dev sentences and computes UAS and LAS."""
    total_tokens = 0
    uas_correct = 0
    las_correct = 0

    sentences_to_eval = dev_sentences[:max_eval]

    t0 = time.time()
    for sent in sentences_to_eval:
        gold_words = sent.words[1:]
        gold_pos = sent.pos_tags[1:]
        gold_heads = sent.heads
        gold_labels = sent.labels

        pred_arcs = parse_sentence(gold_words, gold_pos, classifier)

        # Map dependent -> (head, label)
        pred_map: Dict[int, Tuple[int, str]] = {}
        for head, label, dep in pred_arcs:
            pred_map[dep] = (head, label)

        for dep_id in range(1, len(gold_words) + 1):
            total_tokens += 1
            g_head = gold_heads.get(dep_id, 0)
            g_label = gold_labels.get(dep_id, "dep")

            p_head, p_label = pred_map.get(dep_id, (0, "dep"))

            # UAS: correct head
            if p_head == g_head:
                uas_correct += 1
                # LAS: correct head and correct label
                if p_label == g_label:
                    las_correct += 1

    eval_time = time.time() - t0

    uas = (uas_correct / total_tokens * 100.0) if total_tokens else 0.0
    las = (las_correct / total_tokens * 100.0) if total_tokens else 0.0

    return {
        "uas": uas,
        "las": las,
        "total_tokens": total_tokens,
        "eval_time": eval_time,
        "sentences_evaluated": len(sentences_to_eval),
    }


def format_dependency_tree(words: List[str], pos_tags: List[str], arcs: List[Tuple[int, str, int]]) -> str:
    """Formats dependency arcs into a readable table."""
    pred_map = {dep: (head, label) for head, label, dep in arcs}
    lines = [
        f"{'ID':<4} | {'Word':<15} | {'POS':<8} | {'Head':<6} | {'Head Word':<15} | {'Deprel':<10}",
        "-" * 70,
    ]
    words_with_root = ["<ROOT>"] + list(words)
    for i in range(1, len(words) + 1):
        head, label = pred_map.get(i, (0, "root"))
        head_word = words_with_root[head] if 0 <= head < len(words_with_root) else "<UNK>"
        lines.append(
            f"{i:<4} | {words[i-1]:<15} | {pos_tags[i-1]:<8} | {head:<6} | {head_word:<15} | {label:<10}"
        )
    return "\n".join(lines)


def run_sample_sentences(classifier: Any):
    """Parses and prints the required sample sentences from page 6 of the assignment PDF."""
    print("\n" + "=" * 75)
    print(" SAMPLE SENTENCES (FROM ASSIGNMENT PDF PAGE 6)")
    print("=" * 75)

    samples = [
        (
            "The cat sat on the mat.",
            ["The", "cat", "sat", "on", "the", "mat", "."],
            ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN", "PUNCT"],
        ),
        (
            "She eats a green salad.",
            ["She", "eats", "a", "green", "salad", "."],
            ["PRON", "VERB", "DET", "ADJ", "NOUN", "PUNCT"],
        ),
        (
            "I saw the man with a telescope.",
            ["I", "saw", "the", "man", "with", "a", "telescope", "."],
            ["PRON", "VERB", "DET", "NOUN", "ADP", "DET", "NOUN", "PUNCT"],
        ),
    ]

    for raw_text, words, pos_tags in samples:
        print(f"\nSentence: \"{raw_text}\"")
        arcs = parse_sentence(words, pos_tags, classifier)
        print(format_dependency_tree(words, pos_tags, arcs))


def main():
    print("=" * 75)
    print(" QUESTION 2: TRANSITION-BASED DEPENDENCY PARSER BENCHMARK")
    print("=" * 75)

    train_path = "data/raw/UD_English-EWT/en_ewt-ud-train.conllu"
    dev_path = "data/raw/UD_English-EWT/en_ewt-ud-dev.conllu"

    if not os.path.exists(train_path) or not os.path.exists(dev_path):
        raise FileNotFoundError(
            f"EWT datasets not found at {train_path}. Please ensure UD_English-EWT is cloned."
        )

    # 1. Parse training data
    print("\n>>> 1. Parsing training sentences from CoNLL-U...")
    t0 = time.time()
    train_sentences = parse_conllu(train_path, max_sentences=4000)
    print(f"Loaded {len(train_sentences)} training sentences in {time.time() - t0:.2f}s.")

    # 2. Simulate Oracle
    print("\n>>> 2. Simulating Arc-Standard Oracle to generate training instances...")
    t0 = time.time()
    training_data = []
    for sent in train_sentences:
        instances = simulate_oracle(sent)
        training_data.extend(instances)
    print(f"Generated {len(training_data):,} (configuration, transition) pairs in {time.time() - t0:.2f}s.")

    # 3. Train Classifier
    print("\n>>> 3. Training scikit-learn classifier...")
    t0 = time.time()
    classifier = train_classifier(training_data, max_iter=250)
    train_time = time.time() - t0
    print(f"Classifier trained in {train_time:.2f}s.")

    # 4. Evaluate on Dev Set
    print("\n>>> 4. Parsing and evaluating on en_ewt-ud-dev.conllu...")
    dev_sentences = parse_conllu(dev_path, max_sentences=500)
    metrics = evaluate_dev_set(dev_sentences, classifier, max_eval=500)

    print("\n" + "-" * 75)
    print(f"DEV SET EVALUATION RESULTS ({metrics['sentences_evaluated']} sentences, {metrics['total_tokens']:,} tokens):")
    print("-" * 75)
    print(f"Unlabeled Attachment Score (UAS): {metrics['uas']:.2f}%")
    print(f"Labeled Attachment Score   (LAS): {metrics['las']:.2f}%")
    print(f"Inference Latency:                {metrics['eval_time']:.2f}s ({metrics['eval_time']/metrics['sentences_evaluated']*1000:.1f} ms/sentence)")
    print("-" * 75)

    # 5. Parse required sample sentences
    run_sample_sentences(classifier)

    return metrics


if __name__ == "__main__":
    main()
