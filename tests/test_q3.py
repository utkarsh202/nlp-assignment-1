"""Unit tests for Question 3: Efficient Spelling Corrector."""

import unittest
from src.q3_spelling_corrector.candidate_gen import (
    generate_candidates_edit1,
    preprocess_symmetric_delete,
    generate_candidates_sym_del,
    damerau_levenshtein_distance_1,
)
from src.q3_spelling_corrector.spell_check import (
    build_vocabulary_and_unigram,
    build_bigram_model,
    correct_non_word,
    correct_real_word,
)


class TestQ3SpellingCorrector(unittest.TestCase):
    """Test suite for spelling corrector components."""

    def setUp(self):
        self.toy_corpus = [
            ["I", "have", "a", "good", "feeling", "about", "this"],
            ["This", "is", "a", "test", "sentence"],
            ["I", "would", "like", "to", "see", "the", "world"],
            ["Please", "meet", "me", "at", "the", "station"],
            ["We", "eat", "fresh", "meat", "and", "apples"],
            ["The", "deep", "blue", "sea", "is", "calm"],
        ]
        self.vocab, self.unigram_probs, self.counts = build_vocabulary_and_unigram(self.toy_corpus)
        self.sym_del_dict = preprocess_symmetric_delete(self.vocab)
        self.bigram_model = build_bigram_model(self.toy_corpus)

    def test_damerau_levenshtein_distance(self):
        """Tests edit distance <= 1 detection."""
        self.assertTrue(damerau_levenshtein_distance_1("hello", "hello"))  # distance 0
        self.assertTrue(damerau_levenshtein_distance_1("hello", "helo"))   # deletion
        self.assertTrue(damerau_levenshtein_distance_1("hello", "hellos")) # insertion
        self.assertTrue(damerau_levenshtein_distance_1("hello", "hallo"))  # substitution
        self.assertTrue(damerau_levenshtein_distance_1("hello", "hlelo"))  # transposition
        self.assertFalse(damerau_levenshtein_distance_1("hello", "he"))    # distance > 1

    def test_generate_candidates_edit1(self):
        """Tests Method A candidate generation covers all edit-1 operations."""
        candidates = generate_candidates_edit1("cat")
        self.assertIn("at", candidates)   # deletion
        self.assertIn("act", candidates)  # transposition
        self.assertIn("bat", candidates)  # replacement
        self.assertIn("cats", candidates) # insertion

    def test_symmetric_delete_candidates(self):
        """Tests Method B candidate generation matches expected vocabulary words."""
        candidates = generate_candidates_sym_del("hav", self.sym_del_dict, self.vocab)
        self.assertIn("have", candidates)

        candidates_sent = generate_candidates_sym_del("sentnce", self.sym_del_dict, self.vocab)
        self.assertIn("sentence", candidates_sent)

    def test_correct_non_word(self):
        """Tests non-word error correction using both Method A and Method B."""
        corr_b = correct_non_word("hav", self.vocab, self.unigram_probs, self.sym_del_dict, method="B")
        self.assertEqual(corr_b, "have")

        corr_a = correct_non_word("hav", self.vocab, self.unigram_probs, self.sym_del_dict, method="A")
        self.assertEqual(corr_a, "have")

        # Preservation of capitalization
        corr_cap = correct_non_word("Hav", self.vocab, self.unigram_probs, self.sym_del_dict, method="B")
        self.assertEqual(corr_cap, "Have")

    def test_correct_real_word(self):
        """Tests context-aware real-word error correction."""
        phrase1 = ["I", "would", "like", "to", "sea", "the", "world"]
        fixed1 = correct_real_word(phrase1, self.vocab, self.bigram_model, self.sym_del_dict, threshold=1.0)
        self.assertEqual(fixed1[4], "see")

        phrase2 = ["Please", "meat", "me", "at", "the", "station"]
        fixed2 = correct_real_word(phrase2, self.vocab, self.bigram_model, self.sym_del_dict, threshold=1.0)
        self.assertEqual(fixed2[1], "meet")


if __name__ == "__main__":
    unittest.main()
