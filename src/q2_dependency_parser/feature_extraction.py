"""Feature extraction and model training for transition-based dependency parser."""

import warnings
from typing import Any, Dict, List, Tuple
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from .parser import Configuration


def extract_features(config: Configuration) -> Dict[str, Any]:
    """Extracts features from the current parser configuration.

    Includes the 4 required core assignment features:
    1. POS tag of the word on top of the stack (s0)
    2. POS tag of the second word on the stack (s1)
    3. POS tag of the first word in the buffer (b0)
    4. POS tag of the second word in the buffer (b1)

    Additionally extracts lexical forms and POS combinations for high accuracy.
    """
    s0 = config.stack[-1] if len(config.stack) >= 1 else None
    s1 = config.stack[-2] if len(config.stack) >= 2 else None
    b0 = config.buffer[0] if len(config.buffer) >= 1 else None
    b1 = config.buffer[1] if len(config.buffer) >= 2 else None

    # Required core features
    s0_pos = config.pos_tags[s0] if s0 is not None else "<PAD>"
    s1_pos = config.pos_tags[s1] if s1 is not None else "<PAD>"
    b0_pos = config.pos_tags[b0] if b0 is not None else "<PAD>"
    b1_pos = config.pos_tags[b1] if b1 is not None else "<PAD>"

    features: Dict[str, Any] = {
        # Core 4 assignment features
        "s0_pos": s0_pos,
        "s1_pos": s1_pos,
        "b0_pos": b0_pos,
        "b1_pos": b1_pos,
        # Word forms (lexical)
        "s0_word": config.words[s0].lower() if s0 is not None else "<PAD>",
        "s1_word": config.words[s1].lower() if s1 is not None else "<PAD>",
        "b0_word": config.words[b0].lower() if b0 is not None else "<PAD>",
        "b1_word": config.words[b1].lower() if b1 is not None else "<PAD>",
        # Joint POS pairs
        "s0_s1_pos": f"{s0_pos}_{s1_pos}",
        "s0_b0_pos": f"{s0_pos}_{b0_pos}",
        "s1_b0_pos": f"{s1_pos}_{b0_pos}",
        # Configuration state counters
        "stack_len": min(len(config.stack), 10),
        "buffer_len": min(len(config.buffer), 10),
    }

    return features


def train_classifier(
    training_data: List[Tuple[Configuration, str]],
    max_iter: int = 300,
    C: float = 1.0,
) -> Any:
    """Trains a scikit-learn classifier pipeline to predict transitions from configuration features.

    Args:
        training_data: List of (Configuration, correct_transition) tuples.
        max_iter: Maximum optimization iterations for logistic regression.
        C: Regularization parameter.

    Returns:
        Trained sklearn Pipeline (DictVectorizer + LogisticRegression).
    """
    X_dicts: List[Dict[str, Any]] = []
    y_labels: List[str] = []

    for config, transition in training_data:
        feats = extract_features(config)
        X_dicts.append(feats)
        y_labels.append(transition)

    pipeline = Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=True)),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    solver="lbfgs",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )

    pipeline.fit(X_dicts, y_labels)
    return pipeline
