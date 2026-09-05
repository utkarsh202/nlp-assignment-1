"""Final Passage Analysis and Method Comparison (Question 4 Part 4).

Scores each sentence with:
1. PCFG Constituency Parse log-probability (or 'unparseable')
2. Bigram log-probability / perplexity
3. Trigram log-probability / perplexity

Applies a documented decision rule to determine the chosen method and final verdict.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import nltk
from src.q1_segmentation_tagging.pos_tagging import viterbi_pos_tagging
from src.q4_integrated_editor.pcfg_parser import cky_most_probable_parse, reconcile_tags


def split_into_sentences(text: str) -> List[str]:
    """Splits a passage into sentences using NLTK sent_tokenize or regex."""
    try:
        from nltk.tokenize import sent_tokenize
        sents = sent_tokenize(text)
    except (LookupError, AttributeError):
        try:
            nltk.download("punkt", quiet=True)
            from nltk.tokenize import sent_tokenize
            sents = sent_tokenize(text)
        except Exception:
            sents = re.split(r"(?<=[.?!])\s+", text)

    return [s.strip() for s in sents if s.strip()]


def analyze_final_passage(
    corrected_tokens: List[str],
    pcfg: Any,
    bigram_model: Any,
    trigram_model: Any,
    q1_tagger: Any,
    alerts: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Performs end-of-passage PCFG parsing, bigram/trigram scoring, and method comparison.

    Args:
        corrected_tokens: List of final stream tokens.
        pcfg: Induced PCFG grammar from Penn Treebank.
        bigram_model: Q3 BigramLM model.
        trigram_model: Q1 TrigramLM model.
        q1_tagger: Q1 POS tagger.
        alerts: List of alerts generated during simulation.

    Returns:
        List of per-sentence summary dictionaries.
    """
    full_text = " ".join(corrected_tokens)
    sentences = split_into_sentences(full_text)

    alerts = alerts or []
    summary_rows: List[Dict[str, Any]] = []

    for sent_idx, sent_text in enumerate(sentences):
        # Extract word tokens
        words = re.findall(r"\b[A-Za-z0-9'-]+\b", sent_text)
        if not words:
            continue

        # 1. POS Tagging and Tag Reconciliation
        q1_tagged = viterbi_pos_tagging(words, q1_tagger)
        q1_tags = [tag for _, tag in q1_tagged]
        penn_tags = reconcile_tags(q1_tags)

        # 2. PCFG Constituency Parse
        pcfg_tree, pcfg_log_p, pcfg_status = cky_most_probable_parse(
            words, pcfg, reconciled_tags=penn_tags
        )
        if pcfg_status == "parsed":
            pcfg_result = f"{pcfg_log_p:.2f}"
        elif pcfg_status == "partial_parse":
            pcfg_result = f"{pcfg_log_p:.2f} (partial)"
        else:
            pcfg_result = "unparseable"

        # 3. Bigram Model Score
        bigram_log_p = bigram_model.score_phrase(words)

        # 4. Trigram Model Score
        padded = ["<s>", "<s>"] + [w.lower() for w in words] + ["</s>"]
        trigram_log_p = 0.0
        for i in range(2, len(padded)):
            trigram_log_p += trigram_model.score(padded[i], padded[i - 2], padded[i - 1])

        # 5. Documented Decision Rule:
        # Rule 1 (PCFG First): If PCFG derives a complete S-root parse and average log-prob >= -11.0,
        #         trust PCFG constituency structure -> "Grammatical (Valid Structure)".
        # Rule 2 (Trigram Fluency): If PCFG fails or is unparseable but Trigram LM sequence fluency is high
        #         (average log-prob >= -6.8), trust Trigram LM -> "Grammatical (Fluent Local Sequence)".
        # Rule 3 (Marginal Structure/N-Gram): If partial parse succeeds or Bigram average >= -7.8,
        #         classify as "Marginally Grammatical".
        # Rule 4 (Outlier/Ungrammatical): Otherwise -> "Likely Ungrammatical / Outlier".
        avg_pcfg = pcfg_log_p / len(words) if pcfg_status != "unparseable" else -float("inf")
        avg_trigram = trigram_log_p / len(words)
        avg_bigram = bigram_log_p / len(words)

        if pcfg_status == "parsed" and avg_pcfg >= -11.0:
            chosen_method = "PCFG (Constituency)"
            final_verdict = "Grammatical (Valid Structure)"
        elif avg_trigram >= -6.8:
            chosen_method = "Trigram LM"
            final_verdict = "Grammatical (Fluent Local Sequence)"
        elif pcfg_status == "partial_parse":
            chosen_method = "PCFG (Partial)"
            final_verdict = "Marginally Grammatical"
        elif avg_bigram >= -7.8:
            chosen_method = "Bigram LM"
            final_verdict = "Marginally Grammatical"
        else:
            chosen_method = "Fallback N-Gram"
            final_verdict = "Likely Ungrammatical / Outlier"

        # Count merges and spelling fixes in this sentence text
        merges_in_sent = sum(
            1 for a in alerts if a.get("badge") == "SEGMENT" and any(w.lower() in sent_text.lower() for w in a.get("replacement", []))
        )
        spelling_in_sent = sum(
            1 for a in alerts if a.get("badge") == "SPELL" and str(a.get("replacement", "")).lower() in sent_text.lower()
        )

        summary_rows.append(
            {
                "sentence_idx": sent_idx + 1,
                "sentence_text": sent_text,
                "pcfg_result": pcfg_result,
                "pcfg_tree": pcfg_tree,
                "bigram_score": f"{bigram_log_p:.2f}",
                "trigram_score": f"{trigram_log_p:.2f}",
                "chosen_method": chosen_method,
                "final_verdict": final_verdict,
                "merges_resolved": merges_in_sent,
                "spelling_corrections": spelling_in_sent,
            }
        )

    return summary_rows
