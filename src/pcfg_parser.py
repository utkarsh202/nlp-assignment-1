import math
from collections import defaultdict
import nltk
from nltk import Nonterminal, Tree

UPOS_TO_PTB = {
    "ADJ":"JJ","ADP":"IN","ADV":"RB","AUX":"MD","CCONJ":"CC",
    "DET":"DT","INTJ":"UH","NOUN":"NN","NUM":"CD","PART":"RP",
    "PRON":"PRP","PROPN":"NNP","PUNCT":".","SCONJ":"IN",
    "SYM":"SYM","VERB":"VB","X":"FW"
}

def map_upos_to_ptb(tag):
    return UPOS_TO_PTB.get(tag, "NN")

def train_pcfg():
    """Induces a CNF-binarized PCFG from Penn Treebank, returns grammar."""
    nltk.download("treebank", quiet=True)
    prods = []
    for tree in nltk.corpus.treebank.parsed_sents():
        t = tree.copy(deep=True)
        t.collapse_unary(collapsePOS=False, collapseRoot=False)
        t.chomsky_normal_form(horzMarkov=2)
        prods += t.productions()
    grammar = nltk.induce_pcfg(Nonterminal("S"), prods)
    return grammar

class _CKYParser:
    """Viterbi CKY parser over a CNF PCFG, with POS-tag lexical backoff."""
    def __init__(self, pcfg):
        self.start = pcfg.start()
        self.binary = defaultdict(list)   # (B,C) -> [(A, log_p)]
        self.unary  = defaultdict(list)   # B     -> [(A, log_p)]
        self.lexical = defaultdict(list)  # word  -> [(A, log_p)]
        for prod in pcfg.productions():
            lhs  = prod.lhs()
            rhs  = prod.rhs()
            lp   = math.log(prod.prob()) if prod.prob() > 0 else -100.0
            if len(rhs) == 2:
                self.binary[(rhs[0], rhs[1])].append((lhs, lp))
            elif len(rhs) == 1:
                if isinstance(rhs[0], Nonterminal):
                    self.unary[rhs[0]].append((lhs, lp))
                else:
                    self.lexical[str(rhs[0]).lower()].append((lhs, lp))

    def _unary_close(self, cell):
        changed = True
        for _ in range(6):
            if not changed: break
            changed = False
            for b, (bs, _) in list(cell.items()):
                for a, lp in self.unary.get(b, []):
                    s = bs + lp
                    if a not in cell or s > cell[a][0]:
                        cell[a] = (s, ("U", b))
                        changed = True

    def parse(self, words, ptb_tags=None):
        """Returns (tree, log_prob, status) where status is 'parsed'/'partial'/'unparseable'."""
        n = len(words)
        chart = [[{} for _ in range(n + 1)] for _ in range(n)]
        for i, w in enumerate(words):
            cell = chart[i][i + 1]
            for lhs, lp in self.lexical.get(w.lower(), []):
                if lhs not in cell or lp > cell[lhs][0]:
                    cell[lhs] = (lp, w)
            if ptb_tags and i < len(ptb_tags):
                nt = Nonterminal(ptb_tags[i])
                if nt not in cell:
                    cell[nt] = (-12.0, w)
            if not cell:
                cell[Nonterminal("NN")] = (-15.0, w)
            self._unary_close(cell)

        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                cell = chart[i][j]
                for k in range(i + 1, j):
                    lc, rc = chart[i][k], chart[k][j]
                    if not lc or not rc:
                        continue
                    for b, (bs, _) in lc.items():
                        for c, (cs, _) in rc.items():
                            for a, lp in self.binary.get((b, c), []):
                                s = lp + bs + cs
                                if a not in cell or s > cell[a][0]:
                                    cell[a] = (s, (k, b, c))
                self._unary_close(cell)

        top = chart[0][n]

        def build_tree(sym, i, j, depth=0):
            if depth > 40 or sym not in chart[i][j]:
                return Tree(str(sym), ["..."])
            score, bp = chart[i][j][sym]
            if isinstance(bp, str):
                return Tree(str(sym), [bp])
            elif isinstance(bp, tuple) and bp[0] == "U":
                _, child_sym = bp
                return Tree(str(sym), [build_tree(child_sym, i, j, depth + 1)])
            elif isinstance(bp, tuple) and len(bp) == 3:
                k, left_sym, right_sym = bp
                return Tree(str(sym), [
                    build_tree(left_sym, i, k, depth + 1),
                    build_tree(right_sym, k, j, depth + 1)
                ])
            return Tree(str(sym), [str(bp)])

        if self.start in top:
            tree = build_tree(self.start, 0, n)
            return tree, top[self.start][0], "parsed"
        if top:
            best = max(top.keys(), key=lambda s: top[s][0])
            tree = build_tree(best, 0, n)
            return tree, top[best][0], "partial"
        return None, None, "unparseable"

def parse_with_pcfg(pcfg_grammar, sent_words, upos_tags):
    """Parse a sentence; returns (tree_or_None, log_prob_or_None, status_string)."""
    ptb_tags = [map_upos_to_ptb(t) for t in upos_tags]
    parser = _CKYParser(pcfg_grammar)
    tree, log_p, status = parser.parse(sent_words, ptb_tags=ptb_tags)
    return tree, log_p, status
