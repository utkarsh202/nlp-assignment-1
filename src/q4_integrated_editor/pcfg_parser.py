"""Probabilistic Context-Free Grammar (PCFG) Induction, Tag Reconciliation, and CKY Parsing."""

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import nltk
from nltk import Nonterminal, Tree


def train_pcfg(max_trees: int = 400) -> Any:
    """Induces a Chomsky Normal Form (CNF) PCFG from the Penn Treebank sample."""
    try:
        from nltk.corpus import treebank
        trees = treebank.parsed_sents()[:max_trees]
    except (LookupError, AttributeError):
        nltk.download("treebank", quiet=True)
        from nltk.corpus import treebank
        trees = treebank.parsed_sents()[:max_trees]

    productions = []
    for t in trees:
        t_copy = t.copy(deep=True)
        # Binarize and collapse into CNF
        t_copy.collapse_unary(collapsePOS=False, collapseRoot=False)
        t_copy.chomsky_normal_form(horzMarkov=2)
        productions.extend(t_copy.productions())

    S = Nonterminal("S")
    pcfg = nltk.induce_pcfg(S, productions)
    return pcfg


BROWN_TO_PENN_MAP = {
    "AT": "DT",
    "NN": "NN",
    "NNS": "NNS",
    "NP": "NNP",
    "NP$": "NNP",
    "NP-TL": "NNP",
    "NPS": "NNPS",
    "NPS-TL": "NNPS",
    "NR": "NN",
    "JJ": "JJ",
    "JJ-TL": "JJ",
    "JJR": "JJR",
    "JJS": "JJS",
    "RB": "RB",
    "RB-TL": "RB",
    "RBR": "RBR",
    "RBS": "RBS",
    "VB": "VB",
    "VBD": "VBD",
    "VBG": "VBG",
    "VBN": "VBN",
    "VBP": "VBP",
    "VBZ": "VBZ",
    "IN": "IN",
    "IN-TL": "IN",
    "CS": "IN",
    "CC": "CC",
    "CC-TL": "CC",
    "CD": "CD",
    "CD-TL": "CD",
    "PP$": "PRP$",
    "PPS": "PRP",
    "PPSS": "PRP",
    "PPO": "PRP",
    "PN": "NN",
    "QL": "RB",
    "MD": "MD",
    "BE": "VB",
    "BED": "VBD",
    "BEDZ": "VBD",
    "BEG": "VBG",
    "BEN": "VBN",
    "BER": "VBP",
    "BEZ": "VBZ",
    "DO": "VB",
    "DOD": "VBD",
    "DOZ": "VBZ",
    "HV": "VB",
    "HVD": "VBD",
    "HVZ": "VBZ",
    "WDT": "WDT",
    "WP": "WP",
    "WP$": "WP$",
    "WRB": "WRB",
    ".": ".",
    ",": ",",
    ":": ":",
    "(": "-LRB-",
    ")": "-RRB-",
    "``": "``",
    "''": "''",
}


def reconcile_tags(q1_tags: List[str]) -> List[str]:
    """Maps Q1 Brown Corpus feature-based tags to Penn Treebank tags."""
    reconciled: List[str] = []
    penn_valid = {
        "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD",
        "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR",
        "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
        "WDT", "WP", "WP$", "WRB", ".", ",", ":",
    }

    for tag in q1_tags:
        clean_tag = tag.split("-")[0] if "-" in tag and tag not in BROWN_TO_PENN_MAP else tag
        if clean_tag in BROWN_TO_PENN_MAP:
            reconciled.append(BROWN_TO_PENN_MAP[clean_tag])
        elif tag in penn_valid:
            reconciled.append(tag)
        elif clean_tag in penn_valid:
            reconciled.append(clean_tag)
        else:
            # Fallback for unknown tag forms
            reconciled.append("NN")

    return reconciled


class CKYParser:
    """A robust Viterbi / CKY Parser with lexical backoff for unparseable handling."""

    def __init__(self, pcfg: Any):
        self.pcfg = pcfg
        self.start = pcfg.start()

        self.binary_rules: Dict[Tuple[Nonterminal, Nonterminal], List[Tuple[Nonterminal, float]]] = defaultdict(list)
        self.unary_rules: Dict[Nonterminal, List[Tuple[Nonterminal, float]]] = defaultdict(list)
        self.lexical_rules: Dict[str, List[Tuple[Nonterminal, float]]] = defaultdict(list)

        for prod in pcfg.productions():
            lhs = prod.lhs()
            prob = prod.prob()
            log_p = math.log(prob) if prob > 0 else -100.0

            rhs = prod.rhs()
            if len(rhs) == 2:
                self.binary_rules[(rhs[0], rhs[1])].append((lhs, log_p))
            elif len(rhs) == 1:
                if isinstance(rhs[0], Nonterminal):
                    self.unary_rules[rhs[0]].append((lhs, log_p))
                else:
                    self.lexical_rules[str(rhs[0]).lower()].append((lhs, log_p))

    def _apply_unary_closure(self, cell: Dict[Nonterminal, Tuple[float, Any]]) -> None:
        """Applies unary rules A -> B iteratively to update the cell."""
        changed = True
        iterations = 0
        while changed and iterations < 5:
            changed = False
            iterations += 1
            for b_sym, (b_score, b_bp) in list(cell.items()):
                if b_sym in self.unary_rules:
                    for a_sym, rule_log_p in self.unary_rules[b_sym]:
                        new_score = b_score + rule_log_p
                        if a_sym not in cell or new_score > cell[a_sym][0]:
                            cell[a_sym] = (new_score, ("UNARY", b_sym, b_bp))
                            changed = True

    def parse(
        self,
        words: List[str],
        reconciled_tags: Optional[List[str]] = None,
    ) -> Tuple[Optional[Tree], float, str]:
        """Runs CKY chart parsing to find the most probable parse.

        Returns:
            Tuple of (Tree, log_probability, status).
            status is one of: "parsed", "partial_parse", "unparseable", "empty".
        """
        n = len(words)
        if n == 0:
            return None, -float("inf"), "empty"

        # chart[i][j] maps Nonterminal -> (log_score, backpointer)
        chart: List[List[Dict[Nonterminal, Tuple[float, Any]]]] = [
            [{} for _ in range(n + 1)] for _ in range(n)
        ]

        # Step 1: Lexical terminals (length 1)
        for i in range(n):
            w = words[i].lower()
            matched = False

            if w in self.lexical_rules:
                for lhs, log_p in self.lexical_rules[w]:
                    if lhs not in chart[i][i + 1] or log_p > chart[i][i + 1][lhs][0]:
                        chart[i][i + 1][lhs] = (log_p, words[i])
                        matched = True

            # Use reconciled POS tag for lexical backoff
            if reconciled_tags and i < len(reconciled_tags):
                tag_nt = Nonterminal(reconciled_tags[i])
                if tag_nt not in chart[i][i + 1]:
                    chart[i][i + 1][tag_nt] = (-12.0, words[i])
                    matched = True

            if not matched:
                chart[i][i + 1][Nonterminal("NN")] = (-15.0, words[i])

            self._apply_unary_closure(chart[i][i + 1])

        # Step 2: Binary combinations for length 2 .. n
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                for k in range(i + 1, j):
                    left_cell = chart[i][k]
                    right_cell = chart[k][j]

                    if not left_cell or not right_cell:
                        continue

                    for b_sym, (b_score, _) in left_cell.items():
                        for c_sym, (c_score, _) in right_cell.items():
                            pair = (b_sym, c_sym)
                            if pair in self.binary_rules:
                                for a_sym, rule_log_p in self.binary_rules[pair]:
                                    score = rule_log_p + b_score + c_score
                                    existing = chart[i][j].get(a_sym)
                                    if existing is None or score > existing[0]:
                                        chart[i][j][a_sym] = (score, (k, b_sym, c_sym))

                self._apply_unary_closure(chart[i][j])

        # Step 3: Extract tree from root
        target_root = self.start
        top_cell = chart[0][n]

        def build_tree(sym: Nonterminal, i: int, j: int) -> Tree:
            score, bp = chart[i][j][sym]
            if isinstance(bp, str):
                return Tree(str(sym), [bp])
            elif bp[0] == "UNARY":
                _, child_sym, _ = bp
                return Tree(str(sym), [build_tree(child_sym, i, j)])
            else:
                k, left_sym, right_sym = bp
                return Tree(str(sym), [build_tree(left_sym, i, k), build_tree(right_sym, k, j)])

        # Check for full S derivation
        if target_root in top_cell:
            log_prob = top_cell[target_root][0]
            tree = build_tree(target_root, 0, n)
            return tree, log_prob, "parsed"

        # Check for other valid clause/phrase roots (e.g. SINV, SQ, NP, VP)
        if top_cell:
            best_sym = max(top_cell.keys(), key=lambda s: top_cell[s][0])
            log_prob = top_cell[best_sym][0]
            tree = build_tree(best_sym, 0, n)
            return tree, log_prob, "partial_parse"

        return None, -float("inf"), "unparseable"


def cky_most_probable_parse(
    sentence: List[str],
    pcfg: Any,
    reconciled_tags: Optional[List[str]] = None,
) -> Tuple[Optional[Tree], float, str]:
    """Wraps CKY parsing to return the most probable parse or gracefully report unparseable."""
    parser = CKYParser(pcfg)
    return parser.parse(sentence, reconciled_tags=reconciled_tags)
