"""Comprehensive Evaluation and Benchmark Runner for Question 4.

Includes:
1. 1,000-word Speed Demon benchmark comparing per-token (Segment + Spell) vs periodic grammar triggers (N=5).
2. End-to-end passage comparative analysis with PCFG, Bigram, and Trigram scoring tables.
3. Alert agreement vs final verdict analysis.
"""

import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure workspace root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import nltk

from src.q1_segmentation_tagging.pos_tagging import train_pos_tagger
from src.q1_segmentation_tagging.segmentation import train_trigram_lm
from src.q3_spelling_corrector.candidate_gen import preprocess_symmetric_delete
from src.q3_spelling_corrector.spell_check import (
    build_bigram_model,
    build_vocabulary_and_unigram,
)
from src.q4_integrated_editor.passage_analysis import analyze_final_passage
from src.q4_integrated_editor.pcfg_parser import train_pcfg
from src.q4_integrated_editor.typing_simulation import (
    LiveEditorEngine,
    simulate_fast_typing_merges,
)


def load_evaluation_models() -> Dict[str, Any]:
    """Loads and trains all models on Brown Corpus and Penn Treebank."""
    print("Loading Brown Corpus and Penn Treebank...")
    try:
        from nltk.corpus import brown
        tagged_sents = brown.tagged_sents()[:40000]
    except (LookupError, AttributeError):
        nltk.download("brown", quiet=True)
        from nltk.corpus import brown
        tagged_sents = brown.tagged_sents()[:40000]

    train_words = [[w for w, _ in s] for s in tagged_sents]

    print("Training Q1 Trigram LM and POS Tagger...")
    q1_lm = train_trigram_lm(train_words)
    q1_tagger = train_pos_tagger(tagged_sents)

    print("Building Q3 Vocabulary, Unigrams, Bigrams, and SymSpell index...")
    q3_vocab, q3_unigram_probs, _ = build_vocabulary_and_unigram(train_words)
    q3_sym_del = preprocess_symmetric_delete(q3_vocab)
    q3_bigram = build_bigram_model(train_words)

    print("Inducing Q4 Penn Treebank PCFG (max 350 trees)...")
    pcfg = train_pcfg(max_trees=350)

    print("All models initialized successfully!\n")
    return {
        "q1_lm": q1_lm,
        "q1_tagger": q1_tagger,
        "q3_vocab": q3_vocab,
        "q3_unigram_probs": q3_unigram_probs,
        "q3_sym_del": q3_sym_del,
        "q3_bigram": q3_bigram,
        "pcfg": pcfg,
    }


def get_1000_word_test_corpus() -> List[str]:
    """Fetches exactly 1,000 words from the held-out Brown corpus."""
    from nltk.corpus import brown
    held_out_words = []
    # Use sentences from index 40000 onwards (test split)
    for s in brown.sents()[40000:]:
        held_out_words.extend(s)
        if len(held_out_words) >= 1000:
            break
    return held_out_words[:1000]


def run_speed_demon_benchmark(models: Dict[str, Any]) -> Dict[str, Any]:
    """Runs the 1,000-word Speed Demon benchmark comparing per-token vs full pipeline."""
    print("=" * 70)
    print("SPEED DEMON BENCHMARK: 1,000 WORDS REAL-TIME LATENCY TEST")
    print("=" * 70)

    words = get_1000_word_test_corpus()
    raw_text = " ".join(words)

    # Inject realistic typing noise: 8% merges and occasional typos
    random.seed(42)
    merged_tokens = simulate_fast_typing_merges(raw_text, p=0.08, seed=42)

    # Condition 1: Per-Token Only (Segmentation + Spelling, trigger_n = 999999)
    print(f"Condition 1: Running Per-Token Pipeline (Segmentation + Spelling ONLY) on {len(merged_tokens)} tokens...")
    engine_c1 = LiveEditorEngine(
        q1_lm=models["q1_lm"],
        q1_tagger=models["q1_tagger"],
        q3_vocab=models["q3_vocab"],
        q3_unigram_probs=models["q3_unigram_probs"],
        q3_sym_del_dict=models["q3_sym_del"],
        q3_bigram_model=models["q3_bigram"],
        trigger_n=999999,  # Effectively disabled
    )

    t0_c1 = time.perf_counter()
    for tok in merged_tokens:
        engine_c1.process_incoming_token(tok)
    t1_c1 = time.perf_counter()
    total_time_c1 = (t1_c1 - t0_c1) * 1000.0
    latency_per_word_c1 = total_time_c1 / len(merged_tokens)
    stats_c1 = engine_c1.get_latency_stats()

    # Condition 2: Full Pipeline (Segmentation + Spelling + Grammar Triggers at N=5)
    print(f"Condition 2: Running Full Pipeline (Segment + Spell + Grammar every N=5) on {len(merged_tokens)} tokens...")
    engine_c2 = LiveEditorEngine(
        q1_lm=models["q1_lm"],
        q1_tagger=models["q1_tagger"],
        q3_vocab=models["q3_vocab"],
        q3_unigram_probs=models["q3_unigram_probs"],
        q3_sym_del_dict=models["q3_sym_del"],
        q3_bigram_model=models["q3_bigram"],
        trigger_n=5,
    )

    t0_c2 = time.perf_counter()
    for tok in merged_tokens:
        engine_c2.process_incoming_token(tok)
    t1_c2 = time.perf_counter()
    total_time_c2 = (t1_c2 - t0_c2) * 1000.0
    latency_per_word_c2 = total_time_c2 / len(merged_tokens)
    stats_c2 = engine_c2.get_latency_stats()

    print("\n" + "-" * 70)
    print("SPEED DEMON BENCHMARK RESULTS")
    print("-" * 70)
    print(f"{'Metric':<35} | {'Condition 1 (Seg+Spell)':<18} | {'Condition 2 (Full Pipeline)'}")
    print("-" * 70)
    print(f"{'Tokens Evaluated':<35} | {len(merged_tokens):<18} | {len(merged_tokens)}")
    print(f"{'Total Processing Time (ms)':<35} | {total_time_c1:<18.2f} | {total_time_c2:.2f}")
    print(f"{'Average Latency per Word (ms)':<35} | {latency_per_word_c1:<18.4f} | {latency_per_word_c2:.4f}")
    print(f"{'Segmentation Merges Fixed':<35} | {stats_c1['merges_resolved']:<18} | {stats_c2['merges_resolved']}")
    print(f"{'Total Alerts Generated':<35} | {len(engine_c1.alerts):<18} | {len(engine_c2.alerts)}")
    print(f"{'Grammar Triggers Executed':<35} | {0:<18} | {stats_c2['total_tokens'] // 5}")
    print(f"{'Avg Grammar Trigger Cost (ms)':<35} | {'N/A':<18} | {stats_c2['avg_grammar_latency_ms']:.4f}")
    print(f"{'Throughput (words/second)':<35} | {1000.0 / latency_per_word_c1:<18.1f} | {1000.0 / latency_per_word_c2:.1f}")
    print("-" * 70)

    is_responsive = latency_per_word_c2 < 15.0
    print(f"Human Typist Latency Threshold (< 15 ms/word): {'PASSED' if is_responsive else 'FAILED'} ({latency_per_word_c2:.4f} ms/word)\n")

    return {
        "tokens_evaluated": len(merged_tokens),
        "c1_total_time_ms": total_time_c1,
        "c1_latency_per_word": latency_per_word_c1,
        "c1_alerts": len(engine_c1.alerts),
        "c2_total_time_ms": total_time_c2,
        "c2_latency_per_word": latency_per_word_c2,
        "c2_alerts": len(engine_c2.alerts),
        "c2_grammar_cost_ms": stats_c2["avg_grammar_latency_ms"],
        "c2_merges_resolved": stats_c2["merges_resolved"],
        "is_responsive": is_responsive,
    }


def run_passage_comparative_eval(models: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Runs comparative passage evaluation on two curated test passages."""
    print("=" * 70)
    print("END-OF-PASSAGE COMPARATIVE ANALYSIS (PCFG vs BIGRAM vs TRIGRAM)")
    print("=" * 70)

    passages = {
        "Passage 1 (Narrative Prose)": (
            "The quick brown fox jumps over the lazy dog. "
            "She eats a green salad every afternoon in the quiet courtyard. "
            "I would like to see the world with my friends. "
            "The committee members arrived early to discuss the urgent proposal."
        ),
        "Passage 2 (Formal Report & Complex Clauses)": (
            "The company announced a significant increase in international sales today. "
            "Several investors expressed strong confidence in the executive leadership team. "
            "Government officials met at the central station to sign the trade agreement. "
            "Economic analysts predicted steady growth throughout the upcoming fiscal year."
        ),
    }

    all_results = {}

    for name, text in passages.items():
        print(f"\nEvaluating {name}...")
        # Simulate typing stream with p = 0.08
        simulated_tokens = simulate_fast_typing_merges(text, p=0.08, seed=101)

        engine = LiveEditorEngine(
            q1_lm=models["q1_lm"],
            q1_tagger=models["q1_tagger"],
            q3_vocab=models["q3_vocab"],
            q3_unigram_probs=models["q3_unigram_probs"],
            q3_sym_del_dict=models["q3_sym_del"],
            q3_bigram_model=models["q3_bigram"],
            trigger_n=5,
        )

        for tok in simulated_tokens:
            engine.process_incoming_token(tok)

        # Run passage analysis
        analysis_rows = analyze_final_passage(
            corrected_tokens=engine.accumulated_tokens,
            pcfg=models["pcfg"],
            bigram_model=models["q3_bigram"],
            trigram_model=models["q1_lm"],
            q1_tagger=models["q1_tagger"],
            alerts=engine.alerts,
        )

        all_results[name] = analysis_rows

        print(f"\n{name} - Comparative Scoring Table:")
        print(f"{'#':<3} | {'PCFG LogP':<14} | {'Bigram LogP':<12} | {'Trigram LogP':<12} | {'Chosen Method':<18} | {'Final Verdict'}")
        print("-" * 90)
        for r in analysis_rows:
            print(
                f"{r['sentence_idx']:<3} | {r['pcfg_result']:<14} | {r['bigram_score']:<12} | "
                f"{r['trigram_score']:<12} | {r['chosen_method']:<18} | {r['final_verdict']}"
            )
            print(f"    Text: {r['sentence_text']}")
            print(f"    Fixed Merges: {r['merges_resolved']}, Spelling Fixes: {r['spelling_corrections']}")

    return all_results["Passage 1 (Narrative Prose)"], all_results["Passage 2 (Formal Report & Complex Clauses)"]


def main():
    models = load_evaluation_models()
    benchmark_stats = run_speed_demon_benchmark(models)
    p1_rows, p2_rows = run_passage_comparative_eval(models)

    print("\n" + "=" * 70)
    print("ALL QUESTION 4 EVALUATIONS COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
