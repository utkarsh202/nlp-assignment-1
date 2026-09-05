"""Error Analysis for Question 1: Confusion Matrix and Error-Source Breakdown.

Separates POS tagging errors into:
1. Segmentation-Induced Errors: The word boundary was incorrectly predicted,
   causing the tagger to operate on an invalid token.
2. Genuine POS Errors: The word boundary was segmented correctly, but the tagger
   assigned the wrong grammatical category.
"""

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


def analyze_pipeline_errors(
    test_sentences: List[List[Tuple[str, str]]],
    lm_model: Any,
    pos_tagger: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluates the full pipeline (segmentation + POS tagging) on test sentences.

    Calculates:
    - Overall pipeline accuracy
    - Word boundary precision/recall/F1
    - Count and percentage of segmentation-induced errors
    - Count and percentage of genuine POS tagging errors
    - Confusion matrix of genuine POS errors
    """
    from .segmentation import viterbi_segmentation
    from .pos_tagging import viterbi_pos_tagging

    total_gold_tokens = 0
    correct_tokens = 0
    segmentation_errors = 0
    genuine_pos_errors = 0

    confusion_matrix: Dict[str, Counter] = defaultdict(Counter)
    all_gold_tags: Set[str] = set()
    all_pred_tags: Set[str] = set()

    for sent in test_sentences:
        if not sent:
            continue

        spaceless = "".join(w for w, _ in sent)
        if not spaceless:
            continue

        # Compute character spans for gold tokens
        gold_spans: List[Tuple[int, int, str, str]] = []
        idx = 0
        for w, tag in sent:
            start = idx
            end = idx + len(w)
            gold_spans.append((start, end, w, tag))
            all_gold_tags.add(tag)
            idx = end

        # Run pipeline
        pred_words = viterbi_segmentation(spaceless, lm_model)
        pred_tagged = viterbi_pos_tagging(pred_words, pos_tagger)

        # Compute character spans for predicted tokens
        pred_span_dict: Dict[Tuple[int, int], Tuple[str, str]] = {}
        idx = 0
        for w, tag in pred_tagged:
            start = idx
            end = idx + len(w)
            pred_span_dict[(start, end)] = (w, tag)
            all_pred_tags.add(tag)
            idx = end

        # Compare gold tokens against predicted tokens
        for start, end, w_gold, t_gold in gold_spans:
            total_gold_tokens += 1
            if (start, end) in pred_span_dict:
                # Word was segmented correctly
                _, t_pred = pred_span_dict[(start, end)]
                if t_pred == t_gold:
                    correct_tokens += 1
                else:
                    genuine_pos_errors += 1
                    confusion_matrix[t_gold][t_pred] += 1
            else:
                # Word boundary was wrong -> segmentation error
                segmentation_errors += 1

    total_errors = segmentation_errors + genuine_pos_errors
    seg_error_pct = (segmentation_errors / total_errors * 100.0) if total_errors > 0 else 0.0
    pos_error_pct = (genuine_pos_errors / total_errors * 100.0) if total_errors > 0 else 0.0
    pipeline_acc = (correct_tokens / total_gold_tokens * 100.0) if total_gold_tokens > 0 else 0.0

    return {
        "total_gold_tokens": total_gold_tokens,
        "correct_tokens": correct_tokens,
        "total_errors": total_errors,
        "segmentation_errors": segmentation_errors,
        "segmentation_error_pct": seg_error_pct,
        "genuine_pos_errors": genuine_pos_errors,
        "genuine_pos_error_pct": pos_error_pct,
        "pipeline_accuracy": pipeline_acc,
        "confusion_matrix": confusion_matrix,
        "gold_tags": sorted(list(all_gold_tags)),
        "pred_tags": sorted(list(all_pred_tags)),
    }


def format_confusion_matrix_table(
    confusion_matrix: Dict[str, Counter],
    top_tags: Optional[List[str]] = None,
    max_tags: int = 10,
) -> str:
    """Formats the confusion matrix as a Markdown table."""
    if not confusion_matrix:
        return "No errors recorded in confusion matrix."

    # Identify most frequent confused tags if not provided
    tag_totals = Counter()
    for g_tag, row in confusion_matrix.items():
        for p_tag, c in row.items():
            tag_totals[g_tag] += c
            tag_totals[p_tag] += c

    if top_tags is None:
        top_tags = [t for t, _ in tag_totals.most_common(max_tags)]

    header = ["Actual \\ Predicted"] + top_tags
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]

    for g_tag in top_tags:
        row = [g_tag]
        for p_tag in top_tags:
            cnt = confusion_matrix[g_tag].get(p_tag, 0)
            row.append(str(cnt))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)
