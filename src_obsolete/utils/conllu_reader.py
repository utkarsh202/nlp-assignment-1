"""Utility for reading Universal Dependencies CoNLL-U datasets."""

import os
from typing import Dict, List, Optional, Tuple


def parse_conllu_sentence(lines: List[str], extend_morphology: bool = False) -> List[Tuple[str, str]]:
    """Parses a list of CoNLL-U lines representing a single sentence into (word, tag) pairs.

    If extend_morphology is True, appends Gender and Number features from FEATS
    to the UPOS tag (e.g. NOUN-Fem-Sing, ADJ-Masc-Plur, DET-Fem-Sing).
    """
    sentence: List[Tuple[str, str]] = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) < 6:
            continue

        token_id = parts[0]
        # Ignore multi-word composite token indicators like '2-3' or empty nodes '1.1'
        if "-" in token_id or "." in token_id:
            continue

        word = parts[1]
        upos = parts[3]
        feats = parts[5]

        tag = upos
        if extend_morphology and feats and feats != "_":
            feat_dict = {}
            for item in feats.split("|"):
                if "=" in item:
                    k, v = item.split("=", 1)
                    feat_dict[k] = v

            gender = feat_dict.get("Gender")
            number = feat_dict.get("Number")

            ext_parts = [upos]
            if gender:
                ext_parts.append(gender[:4])
            if number:
                ext_parts.append(number[:4])

            if len(ext_parts) > 1:
                tag = "-".join(ext_parts)

        sentence.append((word, tag))

    return sentence


def load_conllu_file(
    file_path: str,
    extend_morphology: bool = False,
    max_sentences: Optional[int] = None,
) -> List[List[Tuple[str, str]]]:
    """Loads a CoNLL-U file and returns a list of annotated sentences."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CoNLL-U file not found: {file_path}")

    sentences: List[List[Tuple[str, str]]] = []
    current_lines: List[str] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                if current_lines:
                    sent = parse_conllu_sentence(current_lines, extend_morphology=extend_morphology)
                    if sent:
                        sentences.append(sent)
                        if max_sentences and len(sentences) >= max_sentences:
                            break
                    current_lines = []
            else:
                current_lines.append(line)

        if current_lines and (max_sentences is None or len(sentences) < max_sentences):
            sent = parse_conllu_sentence(current_lines, extend_morphology=extend_morphology)
            if sent:
                sentences.append(sent)

    return sentences
