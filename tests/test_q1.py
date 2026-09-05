"""Unit tests for Question 1: Word Segmentation and POS Tagging."""

import unittest
# pyrefly: ignore [missing-import]
from src.q1_segmentation_tagging.segmentation import (
    TrigramLM,
    train_trigram_lm,
    viterbi_segmentation,
    greedy_longest_match_baseline,
)
# pyrefly: ignore [missing-import]
from src.q1_segmentation_tagging.pos_tagging import (
    train_pos_tagger,
    viterbi_pos_tagging,
    most_frequent_tag_baseline,
    extract_morphological_features,
)


class TestQ1Segmentation(unittest.TestCase):
    """Test suite for Word Segmentation."""

    def setUp(self):
        self.toy_corpus = [
            ["this", "is", "a", "test", "sentence"],
            ["this", "is", "another", "simple", "test"],
            ["natural", "language", "processing", "is", "fun"],
            ["we", "can", "segment", "continuous", "text"],
            ["a", "test", "is", "good"],
        ]
        self.lm = train_trigram_lm(self.toy_corpus)

    def test_trigram_lm_properties(self):
        """Tests that language model learns unigrams, bigrams, and vocabulary."""
        self.assertIn("test", self.lm.vocab)
        self.assertIn("processing", self.lm.vocab)
        # Score should be a finite log probability
        score = self.lm.score("test", "a", "simple")
        self.assertIsInstance(score, float)
        self.assertLessEqual(score, 0.0)

    def test_greedy_baseline_simple(self):
        """Tests greedy longest match on a simple phrase."""
        vocab = {"this", "is", "a", "test"}
        segmented = greedy_longest_match_baseline("thisisatest", vocab)
        self.assertEqual(segmented, ["this", "is", "a", "test"])

    def test_greedy_baseline_fallback(self):
        """Tests greedy longest match handles unmatchable characters."""
        vocab = {"this", "test"}
        # 'xyz' not in vocab
        segmented = greedy_longest_match_baseline("thisxyztest", vocab)
        self.assertEqual(segmented, ["this", "x", "y", "z", "test"])

    def test_viterbi_segmentation_simple(self):
        """Tests Viterbi segmentation recovers the space-separated words."""
        segmented = viterbi_segmentation("thisisatest", self.lm)
        self.assertEqual(segmented, ["this", "is", "a", "test"])

    def test_viterbi_segmentation_compound(self):
        """Tests Viterbi segmentation on natural language processing."""
        segmented = viterbi_segmentation("naturallanguageprocessing", self.lm)
        self.assertEqual(segmented, ["natural", "language", "processing"])

    def test_viterbi_empty_input(self):
        """Tests empty text returns empty list."""
        self.assertEqual(viterbi_segmentation("", self.lm), [])
        self.assertEqual(greedy_longest_match_baseline("", self.lm.vocab), [])


class TestQ1POSTagging(unittest.TestCase):
    """Test suite for Morphology-Aware POS Tagging."""

    def setUp(self):
        self.toy_pos_corpus = [
            [("The", "DET"), ("quick", "ADJ"), ("fox", "NOUN"), ("jumps", "VERB")],
            [("A", "DET"), ("dog", "NOUN"), ("barks", "VERB"), ("loudly", "ADV")],
            [("The", "DET"), ("brown", "ADJ"), ("dog", "NOUN"), ("is", "VERB"), ("running", "VERB")],
            [("She", "PRON"), ("reads", "VERB"), ("a", "DET"), ("good", "ADJ"), ("book", "NOUN")],
            [("He", "PRON"), ("drives", "VERB"), ("very", "ADV"), ("quickly", "ADV")],
        ]
        self.tagger = train_pos_tagger(self.toy_pos_corpus)

    def test_extract_morphological_features(self):
        """Tests morphological feature extraction."""
        feats = extract_morphological_features("quickly")
        self.assertIn("suf_ly", feats)
        self.assertIn("pref_qu", feats)
        self.assertIn("shape_all_lower", feats)

        capital_feats = extract_morphological_features("Apple")
        self.assertIn("shape_capitalized", capital_feats)

        digit_feats = extract_morphological_features("2026")
        self.assertIn("shape_has_digit", digit_feats)

    def test_most_frequent_tag_baseline(self):
        """Tests most frequent tag baseline."""
        words = ["The", "dog", "barks"]
        tagged = most_frequent_tag_baseline(words, self.toy_pos_corpus)
        self.assertEqual(tagged, [("The", "DET"), ("dog", "NOUN"), ("barks", "VERB")])

    def test_viterbi_pos_known_words(self):
        """Tests Viterbi POS tagging on known sentences."""
        words = ["The", "quick", "dog", "jumps"]
        tagged = viterbi_pos_tagging(words, self.tagger)
        expected = [("The", "DET"), ("quick", "ADJ"), ("dog", "NOUN"), ("jumps", "VERB")]
        self.assertEqual(tagged, expected)

    def test_viterbi_pos_morphology_unseen_adverb(self):
        """Tests that an unseen word ending in -ly is tagged as ADV using morphology."""
        words = ["The", "fox", "jumps", "swiftly"]
        tagged = viterbi_pos_tagging(words, self.tagger)
        # 'swiftly' was never in the training corpus, but has suffix -ly
        self.assertEqual(tagged[-1][0], "swiftly")
        self.assertEqual(tagged[-1][1], "ADV")

    def test_viterbi_pos_empty(self):
        """Tests empty input returns empty list."""
        self.assertEqual(viterbi_pos_tagging([], self.tagger), [])
        self.assertEqual(most_frequent_tag_baseline([], self.toy_pos_corpus), [])


if __name__ == "__main__":
    unittest.main()
