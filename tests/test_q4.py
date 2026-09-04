"""Unit tests for Question 4: Integrated Background Editor."""

import unittest
from typing import List

import nltk
from nltk import Nonterminal, PCFG

from src.q1_segmentation_tagging.pos_tagging import (
    train_pos_tagger,
    viterbi_pos_tagging,
)
from src.q1_segmentation_tagging.segmentation import (
    TrigramLM,
    train_trigram_lm,
)
from src.q3_spelling_corrector.candidate_gen import preprocess_symmetric_delete
from src.q3_spelling_corrector.spell_check import (
    BigramLM,
    build_bigram_model,
    build_vocabulary_and_unigram,
)
from src.q4_integrated_editor.passage_analysis import (
    analyze_final_passage,
    split_into_sentences,
)
from src.q4_integrated_editor.pcfg_parser import (
    CKYParser,
    cky_most_probable_parse,
    reconcile_tags,
)
from src.q4_integrated_editor.typing_simulation import (
    LiveEditorEngine,
    simulate_fast_typing_merges,
)


class TestQ4IntegratedEditor(unittest.TestCase):
    """Test suite for integrated real-time editor and passage evaluation."""

    def setUp(self):
        # Toy corpus for Q1 and Q3 components
        self.toy_corpus = [
            ["I", "have", "a", "good", "feeling", "about", "this"],
            ["This", "is", "a", "test", "sentence"],
            ["The", "dog", "chased", "the", "cat", "in", "the", "park"],
            ["We", "will", "meet", "at", "the", "station"],
            ["The", "blue", "sea", "is", "very", "calm"],
        ]
        self.vocab, self.unigram_probs, _ = build_vocabulary_and_unigram(self.toy_corpus)
        self.sym_del_dict = preprocess_symmetric_delete(self.vocab)
        self.bigram_model = build_bigram_model(self.toy_corpus)

        # Mock Trigram LM for Q1
        self.q1_lm = train_trigram_lm(self.toy_corpus)

        # Mock POS tagger
        toy_tagged_corpus = [
            [("The", "AT"), ("dog", "NN"), ("chased", "VBD"), ("the", "AT"), ("cat", "NN")],
            [("in", "IN"), ("the", "AT"), ("park", "NN")],
            [("This", "DT"), ("is", "BEDZ"), ("a", "AT"), ("test", "NN"), ("sentence", "NN")],
        ]
        self.q1_tagger = train_pos_tagger(toy_tagged_corpus)

        # Toy PCFG in CNF
        toy_pcfg_grammar = """
            S -> NP VP [1.0]
            NP -> DT NN [0.7] | PRP [0.3]
            VP -> VBD NP [0.7] | VBZ NP [0.3]
            DT -> 'the' [0.8] | 'a' [0.2]
            NN -> 'dog' [0.5] | 'cat' [0.5]
            VBD -> 'chased' [1.0]
            VBZ -> 'is' [1.0]
            PRP -> 'it' [1.0]
        """
        self.toy_pcfg = PCFG.fromstring(toy_pcfg_grammar)

    def test_simulate_fast_typing_merges(self):
        """Tests that space-dropping occurs probabilistically and deterministically with seed."""
        text = "The dog chased the cat in the park today"

        # p = 0.0 -> no merges
        tokens_p0 = simulate_fast_typing_merges(text, p=0.0)
        self.assertEqual(tokens_p0, text.split())

        # p = 1.0 -> merges every consecutive pair
        tokens_p1 = simulate_fast_typing_merges("one two three four", p=1.0)
        self.assertEqual(tokens_p1, ["onetwo", "threefour"])

        # Deterministic with seed
        tokens_seeded1 = simulate_fast_typing_merges(text, p=0.5, seed=42)
        tokens_seeded2 = simulate_fast_typing_merges(text, p=0.5, seed=42)
        self.assertEqual(tokens_seeded1, tokens_seeded2)
        self.assertLess(len(tokens_seeded1), len(text.split()))

    def test_reconcile_tags(self):
        """Tests mapping from Brown Corpus tagset to Penn Treebank tagset."""
        brown_tags = ["AT", "NN", "BEDZ", "VBD", "IN", "UNKNOWN_TAG"]
        penn_tags = reconcile_tags(brown_tags)

        self.assertEqual(penn_tags[0], "DT")
        self.assertEqual(penn_tags[1], "NN")
        self.assertEqual(penn_tags[2], "VBD")
        self.assertEqual(penn_tags[3], "VBD")
        self.assertEqual(penn_tags[4], "IN")
        self.assertEqual(penn_tags[5], "NN")  # Fallback to NN

    def test_cky_parser_toy_grammar(self):
        """Tests CKY chart parser on standard CNF PCFG grammar."""
        parser = CKYParser(self.toy_pcfg)
        words = ["the", "dog", "chased", "the", "cat"]

        tree, log_p, status = parser.parse(words)
        self.assertEqual(status, "parsed")
        self.assertIsNotNone(tree)
        self.assertIsInstance(tree, nltk.Tree)
        self.assertEqual(tree.label(), "S")
        self.assertLess(log_p, 0.0)

    def test_cky_lexical_backoff(self):
        """Tests that CKY parser handles unknown words using reconciled POS tags."""
        parser = CKYParser(self.toy_pcfg)
        # 'hound' is not in toy grammar lexicon, but we provide reconciled tag 'NN'
        words = ["the", "hound", "chased", "the", "cat"]
        reconciled = ["DT", "NN", "VBD", "DT", "NN"]

        tree, log_p, status = parser.parse(words, reconciled_tags=reconciled)
        self.assertIn(status, ["parsed", "partial_parse"])
        self.assertIsNotNone(tree)

    def test_live_editor_engine_alerts(self):
        """Tests the multi-alert engine for segmentation, spelling, and grammar alerts."""
        engine = LiveEditorEngine(
            q1_lm=self.q1_lm,
            q1_tagger=self.q1_tagger,
            q3_vocab=self.vocab,
            q3_unigram_probs=self.unigram_probs,
            q3_sym_del_dict=self.sym_del_dict,
            q3_bigram_model=self.bigram_model,
            trigger_n=3,
        )

        # 1. Non-word error: "sentnce" -> triggers SPELL-ALERT
        alerts_1 = engine.process_incoming_token("sentnce")
        spell_alerts = [a for a in alerts_1 if a["type"] == "SPELL-ALERT"]
        self.assertGreaterEqual(len(spell_alerts), 1)
        self.assertEqual(spell_alerts[0]["replacement"], "sentence")

        # 2. Add tokens to reach trigger_n = 3
        engine.process_incoming_token("This")
        engine.process_incoming_token("is")

        stats = engine.get_latency_stats()
        self.assertGreater(stats["total_tokens"], 0)
        self.assertGreaterEqual(stats["avg_token_latency_ms"], 0.0)

    def test_split_into_sentences(self):
        """Tests robust sentence splitting."""
        passage = "Hello world! This is sentence two. Is this sentence three? Yes it is."
        sents = split_into_sentences(passage)
        self.assertEqual(len(sents), 4)

    def test_passage_analysis_summary(self):
        """Tests passage analysis table generation."""
        tokens = ["The", "dog", "chased", "the", "cat", ".", "This", "is", "a", "test", "sentence", "."]
        summary = analyze_final_passage(
            corrected_tokens=tokens,
            pcfg=self.toy_pcfg,
            bigram_model=self.bigram_model,
            trigram_model=self.q1_lm,
            q1_tagger=self.q1_tagger,
            alerts=[],
        )

        self.assertGreaterEqual(len(summary), 1)
        first_row = summary[0]
        self.assertIn("sentence_idx", first_row)
        self.assertIn("pcfg_result", first_row)
        self.assertIn("bigram_score", first_row)
        self.assertIn("trigram_score", first_row)
        self.assertIn("chosen_method", first_row)
        self.assertIn("final_verdict", first_row)


if __name__ == "__main__":
    unittest.main()
