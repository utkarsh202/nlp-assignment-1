"""Arc-Standard Transition-Based Dependency Parser.

Implements:
1. Configuration state representation (stack, buffer, arcs).
2. CoNLL-U dataset parsing.
3. Arc-standard oracle simulation generating (configuration, transition) pairs.
4. Parsing loop executing predicted transitions with legal action constraints.
"""

from typing import Any, Dict, List, Optional, Set, Tuple


class Configuration:
    """Represents a parser state with a stack, a buffer, and dependency arcs."""

    def __init__(
        self,
        words: Optional[List[str]] = None,
        pos_tags: Optional[List[str]] = None,
        stack: Optional[List[int]] = None,
        buffer: Optional[List[int]] = None,
        arcs: Optional[List[Tuple[int, str, int]]] = None,
    ):
        self.words = words or ["<ROOT>"]
        self.pos_tags = pos_tags or ["<ROOT>"]
        # stack holds token indices (0 is <ROOT>)
        self.stack: List[int] = list(stack) if stack is not None else [0]
        # buffer holds remaining token indices (1 .. n)
        if buffer is not None:
            self.buffer = list(buffer)
        elif words is not None and len(words) > 1:
            self.buffer = list(range(1, len(words)))
        else:
            self.buffer = []
        # arcs: list of (head, label, dependent)
        self.arcs: List[Tuple[int, str, int]] = list(arcs) if arcs is not None else []

    def is_terminal(self) -> bool:
        """Terminal when buffer is empty and stack has only the root token."""
        return len(self.buffer) == 0 and len(self.stack) <= 1

    def copy(self) -> "Configuration":
        """Creates a shallow copy of configuration state for training/simulation."""
        return Configuration(
            words=self.words,
            pos_tags=self.pos_tags,
            stack=list(self.stack),
            buffer=list(self.buffer),
            arcs=list(self.arcs),
        )

    def is_legal(self, transition: str) -> bool:
        """Checks if a transition string is legal in the current configuration."""
        if transition == "SHIFT":
            return len(self.buffer) > 0

        if transition.startswith("LEFT-ARC"):
            # Requires at least 2 tokens on stack, and s1 cannot be root (0)
            return len(self.stack) >= 2 and self.stack[-2] != 0

        if transition.startswith("RIGHT-ARC"):
            # Requires at least 2 tokens on stack
            if len(self.stack) < 2:
                return False
            # If s1 is root (0), right arc is only legal if it is the last reduction or buffer is empty
            if self.stack[-2] == 0:
                return len(self.buffer) == 0 or len(self.stack) == 2
            return True

        return False

    def apply_transition(self, transition: str) -> None:
        """Applies a transition (SHIFT, LEFT-ARC:label, RIGHT-ARC:label) to mutate state."""
        if transition == "SHIFT":
            if len(self.buffer) > 0:
                self.stack.append(self.buffer.pop(0))

        elif transition.startswith("LEFT-ARC"):
            # Format: LEFT-ARC:label or LEFT-ARC
            parts = transition.split(":", 1)
            label = parts[1] if len(parts) > 1 else "dep"
            if len(self.stack) >= 2 and self.stack[-2] != 0:
                s0 = self.stack[-1]
                s1 = self.stack[-2]
                self.arcs.append((s0, label, s1))
                self.stack.pop(-2)

        elif transition.startswith("RIGHT-ARC"):
            # Format: RIGHT-ARC:label or RIGHT-ARC
            parts = transition.split(":", 1)
            label = parts[1] if len(parts) > 1 else "dep"
            if len(self.stack) >= 2:
                s0 = self.stack[-1]
                s1 = self.stack[-2]
                self.arcs.append((s1, label, s0))
                self.stack.pop(-1)


class ParsedSentence:
    """Container for parsed sentence data from a CoNLL-U file."""

    def __init__(
        self,
        words: List[str],
        pos_tags: List[str],
        heads: Dict[int, int],
        labels: Dict[int, str],
        sentence_id: Optional[str] = None,
        text: Optional[str] = None,
    ):
        # words and pos_tags are 0-indexed where index 0 is <ROOT>, index i is token i
        self.words = words
        self.pos_tags = pos_tags
        # heads: token_id (1..n) -> head_id (0..n)
        self.heads = heads
        # labels: token_id (1..n) -> dependency relation string
        self.labels = labels
        self.sentence_id = sentence_id
        self.text = text

    def __len__(self) -> int:
        return len(self.words) - 1


def parse_conllu(file_path: str, max_sentences: Optional[int] = None) -> List[ParsedSentence]:
    """Parses a CoNLL-U file and returns sentences with POS tags and gold-standard relationships."""
    sentences: List[ParsedSentence] = []

    words = ["<ROOT>"]
    pos_tags = ["<ROOT>"]
    heads: Dict[int, int] = {}
    labels: Dict[int, str] = {}
    sent_id: Optional[str] = None
    text: Optional[str] = None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if len(words) > 1:
                    sentences.append(
                        ParsedSentence(
                            words=words,
                            pos_tags=pos_tags,
                            heads=heads,
                            labels=labels,
                            sentence_id=sent_id,
                            text=text,
                        )
                    )
                    if max_sentences and len(sentences) >= max_sentences:
                        break
                words = ["<ROOT>"]
                pos_tags = ["<ROOT>"]
                heads = {}
                labels = {}
                sent_id = None
                text = None
                continue

            if line.startswith("#"):
                if line.startswith("# sent_id ="):
                    sent_id = line.split("=", 1)[1].strip()
                elif line.startswith("# text ="):
                    text = line.split("=", 1)[1].strip()
                continue

            parts = line.split("\t")
            if len(parts) < 8:
                continue

            token_id_str = parts[0]
            # Skip multi-word tokens (e.g., '2-3') and empty nodes (e.g., '1.1')
            if "-" in token_id_str or "." in token_id_str:
                continue

            token_id = int(token_id_str)
            word = parts[1]
            upos = parts[3]
            try:
                head_id = int(parts[6])
            except ValueError:
                head_id = 0
            deprel = parts[7]

            words.append(word)
            pos_tags.append(upos)
            heads[token_id] = head_id
            labels[token_id] = deprel

    if len(words) > 1 and (max_sentences is None or len(sentences) < max_sentences):
        sentences.append(
            ParsedSentence(
                words=words,
                pos_tags=pos_tags,
                heads=heads,
                labels=labels,
                sentence_id=sent_id,
                text=text,
            )
        )

    return sentences


def simulate_oracle(sentence: ParsedSentence) -> List[Tuple[Configuration, str]]:
    """Simulates the arc-standard parsing process to generate (configuration, correct_transition) training pairs.

    Arc-Standard Invariants:
    1. LEFT-ARC(label): s1 is dependent of s0 (head[s1] == s0). s1 is popped.
    2. RIGHT-ARC(label): s0 is dependent of s1 (head[s0] == s1), and ALL dependents of s0
       in the sentence have already been attached. s0 is popped.
    3. SHIFT: Move the front of buffer to stack.
    """
    config = Configuration(words=sentence.words, pos_tags=sentence.pos_tags)
    training_instances: List[Tuple[Configuration, str]] = []
    attached_dependents: Set[int] = set()

    max_steps = 4 * len(sentence) + 10
    step = 0

    while not config.is_terminal() and step < max_steps:
        step += 1

        s0 = config.stack[-1] if len(config.stack) >= 1 else None
        s1 = config.stack[-2] if len(config.stack) >= 2 else None

        transition: Optional[str] = None

        # Check LEFT-ARC condition
        if len(config.stack) >= 2 and s1 != 0 and s0 is not None:
            if sentence.heads.get(s1) == s0:
                label = sentence.labels.get(s1, "dep")
                transition = f"LEFT-ARC:{label}"

        # Check RIGHT-ARC condition
        if transition is None and len(config.stack) >= 2 and s1 is not None and s0 is not None:
            if sentence.heads.get(s0) == s1:
                # In arc-standard, s0 is popped by RIGHT-ARC, so all dependents of s0 must already be attached
                deps_of_s0 = {k for k, h in sentence.heads.items() if h == s0}
                if deps_of_s0.issubset(attached_dependents):
                    label = sentence.labels.get(s0, "dep")
                    transition = f"RIGHT-ARC:{label}"

        # Otherwise SHIFT if buffer has tokens
        if transition is None:
            if len(config.buffer) > 0:
                transition = "SHIFT"
            elif len(config.stack) >= 2:
                # Buffer is empty but stack still has tokens (e.g. non-projective tree or reduction)
                if s1 == 0:
                    label = sentence.labels.get(s0, "root")
                    transition = f"RIGHT-ARC:{label}"
                else:
                    label = sentence.labels.get(s0, "dep")
                    transition = f"RIGHT-ARC:{label}"
            else:
                break

        # Save snapshot of state and chosen transition
        training_instances.append((config.copy(), transition))

        # Update attached dependents
        if transition.startswith("LEFT-ARC"):
            attached_dependents.add(s1)
        elif transition.startswith("RIGHT-ARC"):
            attached_dependents.add(s0)

        config.apply_transition(transition)

    return training_instances


def parse_sentence(
    words: List[str],
    pos_tags: List[str],
    classifier: Any,
) -> List[Tuple[int, str, int]]:
    """Main parser loop: applies predicted transitions using the classifier to build dependency arcs.

    Args:
        words: List of word tokens (without ROOT).
        pos_tags: List of corresponding POS tags.
        classifier: Trained scikit-learn classifier pipeline.

    Returns:
        List of (head, label, dependent) dependency arcs.
    """
    from .feature_extraction import extract_features

    words_with_root = ["<ROOT>"] + list(words)
    pos_with_root = ["<ROOT>"] + list(pos_tags)
    config = Configuration(words=words_with_root, pos_tags=pos_with_root)

    max_steps = 4 * len(words) + 10
    step = 0

    has_predict_proba = hasattr(classifier, "predict_proba")
    classes = getattr(classifier, "classes_", None)

    while not config.is_terminal() and step < max_steps:
        step += 1

        features = extract_features(config)

        chosen_transition: Optional[str] = None

        if has_predict_proba and classes is not None:
            probas = classifier.predict_proba([features])[0]
            ranked_indices = probas.argsort()[::-1]
            for idx in ranked_indices:
                candidate = classes[idx]
                if config.is_legal(candidate):
                    chosen_transition = candidate
                    break

        if chosen_transition is None:
            pred = classifier.predict([features])[0]
            if config.is_legal(pred):
                chosen_transition = pred
            else:
                # Fallback to a legal action
                if len(config.buffer) > 0:
                    chosen_transition = "SHIFT"
                elif len(config.stack) >= 2:
                    chosen_transition = "RIGHT-ARC:dep"
                else:
                    break

        config.apply_transition(chosen_transition)

    # Post-processing fallback: attach any orphan tokens on stack to ROOT (0)
    attached = {dep for _, _, dep in config.arcs}
    for token_id in range(1, len(words) + 1):
        if token_id not in attached:
            config.arcs.append((0, "root", token_id))

    return config.arcs
