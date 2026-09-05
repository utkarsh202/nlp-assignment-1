"""Simulated Fast Typing and Real-Time Multi-Alert Engine (Question 4 Part 1).

Includes:
- simulate_fast_typing_merges: Drops spaces between consecutive words with probability p.
- LiveEditorEngine: Processes incoming token stream with SEGMENT-ALERT, SPELL-ALERT,
  and GRAMMAR-ALERT checks, recording latency and state.
"""

import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from src.q1_segmentation_tagging.pos_tagging import viterbi_pos_tagging
from src.q1_segmentation_tagging.segmentation import viterbi_segmentation
from src.q3_spelling_corrector.spell_check import (
    correct_non_word,
    correct_real_word,
)


def simulate_fast_typing_merges(text: str, p: float = 0.08, seed: Optional[int] = None) -> List[str]:
    """Simulates fast typing by randomly dropping spaces between words with probability p.

    Args:
        text: Input text passage.
        p: Merge probability (default 0.08).
        seed: Optional random seed for reproducible testing.

    Returns:
        List of simulated typing tokens, some of which may be merged.
    """
    if seed is not None:
        random.seed(seed)

    raw_words = text.strip().split()
    if not raw_words:
        return []

    merged_tokens: List[str] = []
    i = 0
    n = len(raw_words)

    while i < n:
        curr_word = raw_words[i]
        # Check if next word should merge with current word
        if i < n - 1 and random.random() < p:
            next_word = raw_words[i + 1]
            merged = curr_word + next_word
            merged_tokens.append(merged)
            i += 2  # skip next word
        else:
            merged_tokens.append(curr_word)
            i += 1

    return merged_tokens


class LiveEditorEngine:
    """Integrated engine hosting Q1, Q3, and Q4 models for real-time stream alerting."""

    def __init__(
        self,
        q1_lm: Any,
        q1_tagger: Any,
        q3_vocab: set,
        q3_unigram_probs: dict,
        q3_sym_del_dict: dict,
        q3_bigram_model: Any,
        trigger_n: int = 5,
    ):
        self.q1_lm = q1_lm
        self.q1_tagger = q1_tagger
        self.q3_vocab = q3_vocab
        self.q3_unigram_probs = q3_unigram_probs
        self.q3_sym_del_dict = q3_sym_del_dict
        self.q3_bigram_model = q3_bigram_model
        self.trigger_n = trigger_n

        self.reset()

    def reset(self):
        """Resets the engine's internal stream state."""
        self.accumulated_tokens: List[str] = []
        self.raw_stream: List[str] = []
        self.alerts: List[Dict[str, Any]] = []
        self.tokens_processed = 0
        self.total_segment_latency = 0.0
        self.total_spell_latency = 0.0
        self.total_grammar_latency = 0.0
        self.segment_count = 0
        self.grammar_triggers_count = 0

    def process_incoming_token(self, token: str) -> List[Dict[str, Any]]:
        """Processes a newly arrived token through SEGMENT, SPELL, and GRAMMAR checks.

        Returns:
            List of alert dictionaries generated for this token.
        """
        token_alerts: List[Dict[str, Any]] = []
        self.raw_stream.append(token)
        clean_token = "".join(c for c in token if c.isalnum()).lower()
        trailing_punct = "".join(c for c in token if not c.isalnum())

        # -------------------------------------------------------------
        # CHECK 1: [SEGMENT-ALERT]
        # -------------------------------------------------------------
        t0_seg = time.perf_counter()
        active_tokens: List[str] = []

        # If token is not in vocab (or length >= 12) and contains letters
        if clean_token and (clean_token not in self.q3_vocab or len(clean_token) >= 12):
            splits = viterbi_segmentation(clean_token, self.q1_lm)
            # Verify if split produced 2 or more valid vocabulary words (rejecting single non-word letters like 's')
            if (
                len(splits) >= 2
                and all(
                    (len(w) >= 2 or w.lower() in {"a", "i"})
                    and w.lower() in self.q3_vocab
                    for w in splits
                )
            ):
                tagged = viterbi_pos_tagging(splits, self.q1_tagger)
                alert = {
                    "type": "SEGMENT-ALERT",
                    "badge": "SEGMENT",
                    "original": token,
                    "replacement": splits,
                    "tagged": tagged,
                    "message": f"Merged token '{token}' separated into {splits} {tagged}",
                    "token_index": self.tokens_processed,
                }
                token_alerts.append(alert)
                self.alerts.append(alert)
                self.segment_count += 1
                for idx, sp_word in enumerate(splits):
                    if idx == len(splits) - 1 and trailing_punct:
                        active_tokens.append(sp_word + trailing_punct)
                    else:
                        active_tokens.append(sp_word)
            else:
                active_tokens.append(token)
        else:
            active_tokens.append(token)

        self.total_segment_latency += (time.perf_counter() - t0_seg) * 1000.0

        # -------------------------------------------------------------
        # CHECK 2: [SPELL-ALERT]
        # -------------------------------------------------------------
        t0_spell = time.perf_counter()
        spell_checked_tokens: List[str] = []

        for t in active_tokens:
            t_clean = "".join(c for c in t if c.isalnum()).lower()
            t_punct = "".join(c for c in t if not c.isalnum())

            if t_clean and t_clean not in self.q3_vocab:
                fixed_clean = correct_non_word(
                    t_clean,
                    self.q3_vocab,
                    self.q3_unigram_probs,
                    self.q3_sym_del_dict,
                    method="B",
                )
                if fixed_clean != t_clean:
                    replacement = fixed_clean + t_punct
                    alert = {
                        "type": "SPELL-ALERT",
                        "badge": "SPELL",
                        "original": t,
                        "replacement": replacement,
                        "message": f"Non-word error '{t}' corrected to '{replacement}'",
                        "token_index": self.tokens_processed,
                    }
                    token_alerts.append(alert)
                    self.alerts.append(alert)
                    spell_checked_tokens.append(replacement)
                else:
                    spell_checked_tokens.append(t)
            else:
                spell_checked_tokens.append(t)

        self.total_spell_latency += (time.perf_counter() - t0_spell) * 1000.0

        # Add to accumulated stream
        for tok in spell_checked_tokens:
            self.accumulated_tokens.append(tok)
            self.tokens_processed += 1

            # ---------------------------------------------------------
            # CHECK 3: [GRAMMAR-ALERT] (every N words)
            # ---------------------------------------------------------
            if self.tokens_processed % self.trigger_n == 0:
                t0_gram = time.perf_counter()
                self.grammar_triggers_count += 1

                window_size = min(len(self.accumulated_tokens), self.trigger_n + 2)
                window = self.accumulated_tokens[-window_size:]

                # A. Real-word error check
                corrected_window = correct_real_word(
                    window,
                    self.q3_vocab,
                    self.q3_bigram_model,
                    self.q3_sym_del_dict,
                    threshold=2.5,
                )
                for w_orig, w_corr in zip(window, corrected_window):
                    if w_orig != w_corr:
                        alert = {
                            "type": "GRAMMAR-ALERT",
                            "badge": "REAL-WORD",
                            "original": w_orig,
                            "replacement": w_corr,
                            "message": f"Real-word contextual error detected: '{w_orig}' -> '{w_corr}'",
                            "token_index": self.tokens_processed,
                        }
                        token_alerts.append(alert)
                        self.alerts.append(alert)

                # Update window in accumulated tokens if modified
                self.accumulated_tokens[-window_size:] = corrected_window

                # B. Perplexity / Plausibility check
                clean_win = ["".join(c for c in w if c.isalnum()).lower() for w in corrected_window if "".join(c for c in w if c.isalnum())]
                if len(clean_win) >= 2:
                    log_p = self.q3_bigram_model.score_phrase(clean_win)
                    avg_log_p = log_p / len(clean_win)
                    # If local word sequence is highly improbable
                    if avg_log_p < -7.5:
                        ppl = math.exp(-avg_log_p) if avg_log_p > -20.0 else 9999.0
                        alert = {
                            "type": "GRAMMAR-ALERT",
                            "badge": "PERPLEXITY",
                            "original": " ".join(corrected_window),
                            "replacement": None,
                            "message": f"Unusual grammatical sequence detected (perplexity {ppl:.1f}): '{' '.join(corrected_window)}'",
                            "token_index": self.tokens_processed,
                        }
                        token_alerts.append(alert)
                        self.alerts.append(alert)

                self.total_grammar_latency += (time.perf_counter() - t0_gram) * 1000.0

        return token_alerts

    def get_latency_stats(self) -> Dict[str, float]:
        """Returns average latency metrics for per-token and per-trigger checks."""
        avg_seg_spell = (
            (self.total_segment_latency + self.total_spell_latency) / max(self.tokens_processed, 1)
        )
        avg_grammar = (
            self.total_grammar_latency / max(self.grammar_triggers_count, 1)
        )
        return {
            "avg_token_latency_ms": avg_seg_spell,
            "avg_grammar_latency_ms": avg_grammar,
            "total_tokens": self.tokens_processed,
            "merges_resolved": self.segment_count,
            "alerts_count": len(self.alerts),
        }
