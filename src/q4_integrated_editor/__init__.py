"""Integrated Background Editor Module (Question 4).

Combines word segmentation, POS tagging, spelling correction, PCFG parsing,
and n-gram scoring into a real-time typing editor and comparative passage analyzer.
"""

from src.q4_integrated_editor.passage_analysis import (
    analyze_final_passage,
    split_into_sentences,
)
from src.q4_integrated_editor.pcfg_parser import (
    BROWN_TO_PENN_MAP,
    CKYParser,
    cky_most_probable_parse,
    reconcile_tags,
    train_pcfg,
)
from src.q4_integrated_editor.typing_simulation import (
    LiveEditorEngine,
    simulate_fast_typing_merges,
)

__all__ = [
    "train_pcfg",
    "reconcile_tags",
    "CKYParser",
    "cky_most_probable_parse",
    "BROWN_TO_PENN_MAP",
    "simulate_fast_typing_merges",
    "LiveEditorEngine",
    "split_into_sentences",
    "analyze_final_passage",
]
