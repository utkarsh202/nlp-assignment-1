import math
import random
from collections import Counter, defaultdict
import nltk

# ---------------------------------------------------------
# Q4: Tagset Reconciliation
# ---------------------------------------------------------
# Map Q1 UPOS tags to Penn Treebank tags for the PCFG parser.
UPOS_TO_PTB = {
    'ADJ': 'JJ',
    'ADP': 'IN',
    'ADV': 'RB',
    'AUX': 'MD', # or VBP/VBZ depending on tense
    'CCONJ': 'CC',
    'DET': 'DT',
    'INTJ': 'UH',
    'NOUN': 'NN',
    'NUM': 'CD',
    'PART': 'POS',
    'PRON': 'PRP',
    'PROPN': 'NNP',
    'PUNCT': '.',
    'SCONJ': 'IN',
    'SYM': 'SYM',
    'VERB': 'VB',
    'X': 'FW'
}

def map_upos_to_ptb(upos_tag):
    return UPOS_TO_PTB.get(upos_tag, 'NN')

# ---------------------------------------------------------
# Q4: PCFG Parser
# ---------------------------------------------------------
def train_pcfg():
    """Train PCFG from the NLTK Penn Treebank sample."""
    nltk.download('treebank', quiet=True)
    productions = []
    for tree in nltk.corpus.treebank.parsed_sents():
        productions += tree.productions()
    S = nltk.Nonterminal('S')
    grammar = nltk.induce_pcfg(S, productions)
    return nltk.ViterbiParser(grammar)

def parse_with_pcfg(parser, words, pos_tags):
    """
    Attempt to parse a sentence using the PCFG.
    We convert the words into their reconciled POS tags so the parser 
    doesn't fail on unseen words (since the treebank vocabulary is small).
    """
    reconciled_tags = [map_upos_to_ptb(t) for t in pos_tags]
    try:
        # We parse the sequence of tags rather than the words themselves 
        # to ensure coverage, as the PCFG lexicon is limited.
        parses = list(parser.parse(reconciled_tags))
        if parses:
            return parses[0].prob()
        return None
    except ValueError:
        # Usually happens if a tag isn't in the PCFG grammar
        return None

# ---------------------------------------------------------
# Q4: Sentence N-gram Scorers (Add-k Smoothed)
# ---------------------------------------------------------
def train_sentence_lms(sents, k=0.01):
    """Train shared bigram and trigram LMs for sentence scoring."""
    uni = Counter()
    bi = Counter()
    tri = Counter()
    
    for sent in sents:
        words = ["<S>", "<S>"] + [w for w, _ in sent] + ["</S>"]
        for w in words: uni[w] += 1
        for i in range(len(words)-1): bi[(words[i], words[i+1])] += 1
        for i in range(len(words)-2): tri[(words[i], words[i+1], words[i+2])] += 1
        
    vocab_size = len(uni)
    
    def score_sentence_bigram(words):
        w = ["<S>"] + words + ["</S>"]
        log_p = 0.0
        for i in range(len(w)-1):
            c_bi = bi.get((w[i], w[i+1]), 0)
            c_uni = uni.get(w[i], 0)
            prob = (c_bi + k) / (c_uni + k * vocab_size)
            log_p += math.log(prob)
        return log_p

    def score_sentence_trigram(words):
        w = ["<S>", "<S>"] + words + ["</S>"]
        log_p = 0.0
        for i in range(len(w)-2):
            c_tri = tri.get((w[i], w[i+1], w[i+2]), 0)
            c_bi = bi.get((w[i], w[i+1]), 0)
            prob = (c_tri + k) / (c_bi + k * vocab_size)
            log_p += math.log(prob)
        return log_p

    return score_sentence_bigram, score_sentence_trigram

# ---------------------------------------------------------
# Q4: Final Passage Scorer
# ---------------------------------------------------------
def analyze_sentence(words, pos_tags, pcfg_parser, score_bigram, score_trigram):
    """
    Score a sentence across all 3 models and make a final verdict.
    """
    pcfg_prob = parse_with_pcfg(pcfg_parser, words, pos_tags)
    bi_score = score_bigram(words)
    tri_score = score_trigram(words)
    
    # Decision Rule: 
    # Prefer PCFG if it parses (since it checks global grammatical structure).
    # Otherwise, fall back to Trigram (local context), then Bigram.
    if pcfg_prob is not None:
        verdict = "Grammatical (PCFG parsed)"
        chosen = "PCFG"
    else:
        # Simple thresholding for n-grams (can be tuned)
        avg_tri = tri_score / len(words)
        if avg_tri > -8.0:
            verdict = "Grammatical (Trigram threshold passed)"
            chosen = "Trigram"
        else:
            verdict = "Ungrammatical (PCFG failed, N-gram score too low)"
            chosen = "Bigram/Trigram"
            
    return {
        "pcfg_prob": pcfg_prob,
        "bigram_score": bi_score,
        "trigram_score": tri_score,
        "chosen_method": chosen,
        "verdict": verdict
    }
