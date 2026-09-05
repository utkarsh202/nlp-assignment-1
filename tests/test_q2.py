"""Unit tests for Question 2: Transition-Based Dependency Parser."""

import unittest
from src.q2_dependency_parser.parser import (
    Configuration,
    ParsedSentence,
    parse_conllu,
    simulate_oracle,
    parse_sentence,
)
from src.q2_dependency_parser.feature_extraction import (
    extract_features,
    train_classifier,
)


class TestQ2DependencyParser(unittest.TestCase):
    """Test suite for arc-standard dependency parser components."""

    def setUp(self):
        # Synthetic sentence: "The cat sat"
        # 1: The (DET, head=2, det)
        # 2: cat (NOUN, head=3, nsubj)
        # 3: sat (VERB, head=0, root)
        self.words = ["<ROOT>", "The", "cat", "sat"]
        self.pos_tags = ["<ROOT>", "DET", "NOUN", "VERB"]
        self.heads = {1: 2, 2: 3, 3: 0}
        self.labels = {1: "det", 2: "nsubj", 3: "root"}
        self.sent = ParsedSentence(
            words=self.words,
            pos_tags=self.pos_tags,
            heads=self.heads,
            labels=self.labels,
            text="The cat sat",
        )

    def test_configuration_transitions(self):
        """Tests SHIFT, LEFT-ARC, and RIGHT-ARC transition mechanics."""
        config = Configuration(words=self.words, pos_tags=self.pos_tags)
        self.assertEqual(config.stack, [0])
        self.assertEqual(config.buffer, [1, 2, 3])
        self.assertFalse(config.is_terminal())

        # SHIFT: moves 1 to stack
        self.assertTrue(config.is_legal("SHIFT"))
        config.apply_transition("SHIFT")
        self.assertEqual(config.stack, [0, 1])
        self.assertEqual(config.buffer, [2, 3])

        # SHIFT: moves 2 to stack
        config.apply_transition("SHIFT")
        self.assertEqual(config.stack, [0, 1, 2])
        self.assertEqual(config.buffer, [3])

        # LEFT-ARC:det -> head=2, dep=1, pops 1
        self.assertTrue(config.is_legal("LEFT-ARC:det"))
        config.apply_transition("LEFT-ARC:det")
        self.assertEqual(config.stack, [0, 2])
        self.assertEqual(config.arcs, [(2, "det", 1)])

        # SHIFT: moves 3 to stack
        config.apply_transition("SHIFT")
        self.assertEqual(config.stack, [0, 2, 3])

        # LEFT-ARC:nsubj -> head=3, dep=2, pops 2
        config.apply_transition("LEFT-ARC:nsubj")
        self.assertEqual(config.stack, [0, 3])
        self.assertIn((3, "nsubj", 2), config.arcs)

        # RIGHT-ARC:root -> head=0, dep=3, pops 3
        self.assertTrue(config.is_legal("RIGHT-ARC:root"))
        config.apply_transition("RIGHT-ARC:root")
        self.assertEqual(config.stack, [0])
        self.assertEqual(config.buffer, [])
        self.assertTrue(config.is_terminal())
        self.assertIn((0, "root", 3), config.arcs)

    def test_oracle_simulation(self):
        """Tests that oracle simulation generates correct arc-standard transitions."""
        instances = simulate_oracle(self.sent)
        transitions = [trans for _, trans in instances]
        expected_transitions = [
            "SHIFT",
            "SHIFT",
            "LEFT-ARC:det",
            "SHIFT",
            "LEFT-ARC:nsubj",
            "RIGHT-ARC:root",
        ]
        self.assertEqual(transitions, expected_transitions)

    def test_feature_extraction(self):
        """Tests that the 4 required core features (s0_pos, s1_pos, b0_pos, b1_pos) are present."""
        config = Configuration(words=self.words, pos_tags=self.pos_tags)
        config.apply_transition("SHIFT")  # stack=[0, 1], buffer=[2, 3]
        feats = extract_features(config)

        # Check required features
        self.assertIn("s0_pos", feats)
        self.assertIn("s1_pos", feats)
        self.assertIn("b0_pos", feats)
        self.assertIn("b1_pos", feats)

        self.assertEqual(feats["s0_pos"], "DET")
        self.assertEqual(feats["s1_pos"], "<ROOT>")
        self.assertEqual(feats["b0_pos"], "NOUN")
        self.assertEqual(feats["b1_pos"], "VERB")

    def test_classifier_training_and_parsing(self):
        """Tests end-to-end classifier training on synthetic sentences and parsing execution."""
        instances = simulate_oracle(self.sent)
        classifier = train_classifier(instances, max_iter=100)

        # Parse the sentence
        words = ["The", "cat", "sat"]
        pos_tags = ["DET", "NOUN", "VERB"]
        pred_arcs = parse_sentence(words, pos_tags, classifier)

        # Check that 3 dependency arcs are produced and span tokens 1, 2, 3
        dependents = {dep for _, _, dep in pred_arcs}
        self.assertEqual(dependents, {1, 2, 3})


if __name__ == "__main__":
    unittest.main()
