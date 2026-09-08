"""
Question 4 — Integrated NLP Editor (Streamlit App)
Self-contained: all helper functions defined inline.
Run from project root:
    .venv/bin/python -m streamlit run src/streamlit_app.py
"""

import math
import random
import time
from collections import Counter, defaultdict

import nltk
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────
# Page config — must be the very first Streamlit call
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Q4 – Integrated NLP Editor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
}
.sub-header { font-size: 1rem; color: #6B7280; margin-bottom: 1rem; }
.badge-segment {
    background:#FEF3C7; color:#92400E; padding:2px 8px;
    border-radius:4px; font-weight:600; font-size:0.8rem; border:1px solid #FCD34D;
}
.badge-spell {
    background:#FEE2E2; color:#991B1B; padding:2px 8px;
    border-radius:4px; font-weight:600; font-size:0.8rem; border:1px solid #FCA5A5;
}
.badge-grammar {
    background:#EDE9FE; color:#5B21B6; padding:2px 8px;
    border-radius:4px; font-weight:600; font-size:0.8rem; border:1px solid #C4B5FD;
}
.alert-card {
    background:#FFFFFF; border-left:4px solid #3B82F6;
    padding:10px 14px; margin-bottom:8px; border-radius:6px;
    box-shadow:0 1px 3px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════
# SECTION A — Q1: Trigram LM + Viterbi Segmenter + HMM POS
# ═════════════════════════════════════════════════════════

def train_trigram_lm(sents):
    uni, bi, tri = Counter(), Counter(), Counter()
    for sent in sents:
        words = ["<S>", "<S>"] + [w for w, _ in sent] + ["</S>"]
        for w in words: uni[w] += 1
        for i in range(len(words)-1): bi[(words[i], words[i+1])] += 1
        for i in range(len(words)-2): tri[(words[i], words[i+1], words[i+2])] += 1
    return uni, bi, tri


def make_log_prob(uni, bi, tri, vocab_size):
    def log_prob(w1, w2, w3):
        return math.log((tri.get((w1,w2,w3),0)+1.0) / (bi.get((w1,w2),0)+vocab_size))
    return log_prob


def viterbi_segment(text, vocab, log_prob_fn, max_word_len=20):
    n = len(text)
    INF = float("-inf")
    dp = [None]*(n+1)
    dp[0] = {("<S>","<S>"): (0.0, None)}
    for i in range(n):
        if dp[i] is None: continue
        for j in range(i+1, min(i+max_word_len+1, n+1)):
            word = text[i:j]
            if word not in vocab: continue
            if dp[j] is None: dp[j] = {}
            for (p2,p1),(score,_) in dp[i].items():
                ns = score + log_prob_fn(p2, p1, word)
                key = (p1, word)
                if key not in dp[j] or ns > dp[j][key][0]:
                    dp[j][key] = (ns, (i, p2))
    if dp[n] is None: return [text]
    best_score, best_key = INF, None
    for (p2,p1),(score,_) in dp[n].items():
        fs = score + log_prob_fn(p2, p1, "</S>")
        if fs > best_score: best_score, best_key = fs, (p2,p1)
    if best_key is None: return [text]
    words, ci = [], n
    p2, p1 = best_key
    while ci > 0:
        words.append(p1)
        _, back = dp[ci][(p2,p1)]
        ci, prev_p2 = back
        p1, p2 = p2, prev_p2
    words.reverse()
    return words if words else [text]


def train_hmm(sents):
    emission, trans, tag_cnt, tagset = defaultdict(Counter), defaultdict(Counter), Counter(), set()
    for sent in sents:
        tags = ["<S>","<S>"] + [t for _,t in sent] + ["</S>"]
        for w,t in sent: emission[t][w]+=1; tag_cnt[t]+=1; tagset.add(t)
        for i in range(2,len(tags)): trans[(tags[i-2],tags[i-1])][tags[i]]+=1
    return emission, trans, tag_cnt, tagset


def make_hmm_fns(emission, trans, tag_cnt, tagset):
    V = sum(len(v) for v in emission.values())
    N = len(tagset)
    def emit_lp(tag, word):
        return math.log((emission[tag].get(word,0)+1.0)/(tag_cnt.get(tag,0)+V))
    def trans_lp(t1,t2,t3):
        total = sum(trans[(t1,t2)].values())
        return math.log((trans[(t1,t2)].get(t3,0)+1.0)/(total+N)) if total>0 else math.log(1.0/N)
    return emit_lp, trans_lp


def viterbi_pos(words, tagset, emit_lp, trans_lp):
    n = len(words)
    if n == 0: return []
    INF = float("-inf")
    dp = [{} for _ in range(n+1)]
    tags = list(tagset)
    dp[0][("<S>","<S>")] = (0.0, None)
    for i, word in enumerate(words):
        for (t1,t2),(score,_) in dp[i].items():
            for t3 in tags:
                ns = score + emit_lp(t3,word) + trans_lp(t1,t2,t3)
                key = (t2,t3)
                if key not in dp[i+1] or ns > dp[i+1][key][0]:
                    dp[i+1][key] = (ns, t1)
    best_score, best_key = INF, None
    for (t1,t2),(score,_) in dp[n].items():
        fs = score + trans_lp(t1,t2,"</S>")
        if fs > best_score: best_score, best_key = fs, (t1,t2)
    if best_key is None: return ["NOUN"]*n
    pred, ci = [], n
    t1, t2 = best_key
    while ci > 0:
        pred.append(t2)
        _, prev_t1 = dp[ci][(t1,t2)]
        ci -= 1; t1, t2 = prev_t1, t1
    pred.reverse()
    return pred

# ═════════════════════════════════════════════════════════
# SECTION B — Q3: Spelling Corrector
# ═════════════════════════════════════════════════════════

def build_delete_index(vocab):
    di = defaultdict(set)
    for word in vocab:
        for i in range(len(word)):
            di[word[:i]+word[i+1:]].add(word)
    return di


def method_b_candidates(word, delete_index):
    """Return (ed1_candidates, ed2_candidates) for a misspelled word."""
    word = word.lower()
    ed1 = set()
    # 1. Deletion in dictionary word (Insertion in misspelled word)
    ed1.update(delete_index.get(word, set()))
    # 2. Deletion in misspelled word (Deletion in dict word) or Substitution
    for i in range(len(word)):
        deleted1 = word[:i] + word[i+1:]
        ed1.update(delete_index.get(deleted1, set()))
        
    # ed2: delete two chars from word, look up in index
    ed2 = set()
    for i in range(len(word)):
        d1 = word[:i] + word[i+1:]
        for j in range(len(d1)):
            d2 = d1[:j] + d1[j+1:]
            ed2.update(delete_index.get(d2, set()))
            
    ed2 -= ed1  # keep only pure ed2 candidates
    return ed1, ed2


def best_candidate(word, delete_index, word_counts):
    """Return the best spelling correction candidate, or None if word is already valid."""
    ed1, ed2 = method_b_candidates(word, delete_index)
    if ed1:
        return max(ed1, key=lambda w: word_counts.get(w, 0))
    if ed2:
        return max(ed2, key=lambda w: word_counts.get(w, 0))
    return None

# ═════════════════════════════════════════════════════════
# SECTION C — Q4: Tagset Reconciliation + PCFG + N-gram
# ═════════════════════════════════════════════════════════

UPOS_TO_PTB = {
    "ADJ":"JJ","ADP":"IN","ADV":"RB","AUX":"MD","CCONJ":"CC",
    "DET":"DT","INTJ":"UH","NOUN":"NN","NUM":"CD","PART":"RP",
    "PRON":"PRP","PROPN":"NNP","PUNCT":".","SCONJ":"IN",
    "SYM":"SYM","VERB":"VB","X":"FW"
}

def map_upos_to_ptb(tag):
    return UPOS_TO_PTB.get(tag, "NN")


def train_pcfg(max_trees=500):
    nltk.download("treebank", quiet=True)
    prods = []
    for tree in nltk.corpus.treebank.parsed_sents()[:max_trees]:
        prods += tree.productions()
    grammar = nltk.induce_pcfg(nltk.Nonterminal("S"), prods)
    return nltk.ViterbiParser(grammar)


def parse_with_pcfg(parser, pos_tags):
    ptb = [map_upos_to_ptb(t) for t in pos_tags]
    try:
        parses = list(parser.parse(ptb))
        if parses: return parses[0].prob(), parses[0]
    except Exception: pass
    return None, None


def train_sentence_lms(sents, k=0.01):
    uni, bi, tri = Counter(), Counter(), Counter()
    for sent in sents:
        words = ["<S>","<S>"] + [w for w,_ in sent] + ["</S>"]
        for w in words: uni[w] += 1
        for i in range(len(words)-1): bi[(words[i],words[i+1])] += 1
        for i in range(len(words)-2): tri[(words[i],words[i+1],words[i+2])] += 1
    V = len(uni)
    def score_bigram(words):
        w = ["<S>"]+words+["</S>"]
        return sum(math.log((bi.get((w[i],w[i+1]),0)+k)/(uni.get(w[i],0)+k*V)) for i in range(len(w)-1))
    def score_trigram(words):
        w = ["<S>","<S>"]+words+["</S>"]
        return sum(math.log((tri.get((w[i],w[i+1],w[i+2]),0)+k)/(bi.get((w[i],w[i+1]),0)+k*V)) for i in range(len(w)-2))
    def score_phrase_bigram(words):  # used for grammar alerting
        if len(words) < 2: return 0.0
        return score_bigram(words) / len(words)
    return score_bigram, score_trigram, score_phrase_bigram

# ═════════════════════════════════════════════════════════
# SECTION D — Typing Simulation Engine (from old code)
# ═════════════════════════════════════════════════════════

def simulate_fast_typing_merges(text, p=0.08, seed=None):
    """Randomly drop spaces between words with probability p."""
    if seed is not None: random.seed(seed)
    raw = text.strip().split()
    if not raw: return []
    merged, i = [], 0
    while i < len(raw):
        if i < len(raw)-1 and random.random() < p:
            merged.append(raw[i] + raw[i+1]); i += 2
        else:
            merged.append(raw[i]); i += 1
    return merged


def process_token_with_alerts(token, vocab, log_prob_fn, delete_index, uni_counts,
                               accumulated, score_phrase_bigram, trigger_n,
                               grammar_threshold, seg_count, spell_count, tokens_processed):
    """Process one token through Segment → Spell → Grammar checks."""
    alerts = []

    # Separate alphabetic content from trailing punctuation (e.g. "project." → "project" + ".")
    # Preserve punctuation tokens like "." so sentence-splitting works downstream.
    stripped = token.rstrip(".,!?;:")
    trail_punct = token[len(stripped):]
    clean = "".join(c for c in stripped if c.isalnum()).lower()

    # ── SEGMENT-ALERT ──────────────────────────────────────
    # Only attempt segmentation when the token is:
    #   (a) not in vocab, AND
    #   (b) long enough to plausibly be two merged words (>= 8 chars)
    # This prevents misspellings like 'hav' being fed to the segmenter.
    t0 = time.perf_counter()
    active = []
    if clean and clean not in vocab and len(clean) >= 8:
        splits = viterbi_segment(clean, vocab, log_prob_fn)
        # Accept split only if:
        #   • at least 2 sub-words
        #   • every sub-word is in vocab and has length >= 2 (or is 'a'/'i')
        #   • the shortest sub-word is at least 3 chars (avoids junk splits)
        if (len(splits) >= 2
                and all(w in vocab and (len(w) >= 3 or w in {"a", "i", "an"}) for w in splits)):
            alerts.append({
                "type":"SEGMENT-ALERT", "badge":"SEGMENT",
                "original": token,
                "message": f"Merged token '{token}' → {' | '.join(splits)}"
            })
            seg_count += 1
            active.extend(splits)
        else:
            active.append(clean)
    else:
        active.append(clean if clean else token)
    seg_lat = (time.perf_counter()-t0)*1000

    # ── SPELL-ALERT ────────────────────────────────────────
    # Skip very short tokens (1-2 chars) — they are almost always valid
    # abbreviations or single-letter words ('a', 'i', etc.)
    t0 = time.perf_counter()
    spell_checked = []
    for t in active:
        t_clean = t.lower()
        if t_clean and len(t_clean) >= 3 and t_clean not in vocab:
            best = best_candidate(t_clean, delete_index, uni_counts)
            if best and best != t_clean:
                alerts.append({
                    "type":"SPELL-ALERT","badge":"SPELL",
                    "original": t,
                    "message": f"Non-word '{t}' → corrected to '{best}'"
                })
                spell_count += 1
                spell_checked.append(best)
            else:
                spell_checked.append(t_clean)
        else:
            spell_checked.append(t_clean if t_clean else t)
    spell_lat = (time.perf_counter()-t0)*1000

    # Append corrected words; also append trailing punctuation as a separate token
    # so that analyze_final_passage can split on '.', '!', '?', ';'
    for tok in spell_checked:
        if tok:  # skip empty strings
            accumulated.append(tok)
            tokens_processed += 1

            # ── GRAMMAR-ALERT every N tokens ──────────────────
            if tokens_processed % trigger_n == 0:
                t0 = time.perf_counter()
                window_size = min(len(accumulated), trigger_n + 2)
                window = [w for w in accumulated[-window_size:] if w.isalpha()]
                if len(window) >= 2:
                    avg_log_p = score_phrase_bigram(window)
                    if avg_log_p < grammar_threshold:
                        ppl = math.exp(-avg_log_p) if avg_log_p > -20 else 9999.0
                        alerts.append({
                            "type":"GRAMMAR-ALERT","badge":"GRAMMAR",
                            "original": " ".join(window),
                        "message": f"Unusual sequence (perplexity≈{ppl:.1f}): '{ ' '.join(window)}'"
                    })

    return alerts, accumulated, seg_count, spell_count, tokens_processed, seg_lat, spell_lat


def analyze_final_passage(accumulated, pos_tags, pcfg_parser, score_bi, score_tri):
    """Split accumulated stream into sentences, score each across all 3 models."""
    rows = []
    current, cur_tags = [], []
    for w, t in zip(accumulated, pos_tags):
        current.append(w); cur_tags.append(t)
        if w in {".", "!", "?", ";"}:
            rows.append((current[:], cur_tags[:])); current, cur_tags = [], []
    if current: rows.append((current, cur_tags))
    if not rows: rows = [(accumulated, pos_tags)]

    results = []
    for i, (sent_words, sent_tags) in enumerate(rows):
        pcfg_prob, pcfg_tree = parse_with_pcfg(pcfg_parser, sent_tags)
        bi_score  = score_bi(sent_words)
        tri_score = score_tri(sent_words)
        n = max(len(sent_words), 1)
        avg_tri = tri_score / n

        if pcfg_prob is not None:
            chosen  = "PCFG"
            verdict = "Grammatical (PCFG parsed)"
        elif avg_tri > -9.0:
            chosen  = "Trigram"
            verdict = "Grammatical (Trigram OK)"
        else:
            chosen  = "Bigram/Trigram"
            verdict = "Possibly Ungrammatical"

        results.append({
            "sentence_idx":  i+1,
            "sentence_text": " ".join(sent_words),
            "pcfg_result":   f"{pcfg_prob:.2e}" if pcfg_prob else "Unparseable",
            "pcfg_tree":     pcfg_tree,
            "bigram_score":  f"{bi_score:.2f}",
            "trigram_score": f"{tri_score:.2f}",
            "chosen_method": chosen,
            "final_verdict": verdict,
        })
    return results


# ═════════════════════════════════════════════════════════
# MODEL LOADING (cached — runs once, shows progress)
# ═════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_all_models():
    nltk.download("brown",    quiet=True)
    nltk.download("treebank", quiet=True)
    sents = nltk.corpus.brown.tagged_sents(tagset="universal")[:20000]

    uni, bi, tri   = train_trigram_lm(sents)
    vocab          = set(uni.keys()) - {"<S>","</S>"}
    log_prob_fn    = make_log_prob(uni, bi, tri, len(vocab))

    hmm_emit, hmm_trans, hmm_tag_cnt, hmm_tagset = train_hmm(sents)
    emit_lp, trans_lp = make_hmm_fns(hmm_emit, hmm_trans, hmm_tag_cnt, hmm_tagset)

    del_idx        = build_delete_index(vocab)
    score_bi, score_tri, score_phrase = train_sentence_lms(sents)
    pcfg_parser    = train_pcfg(max_trees=500)

    return (vocab, log_prob_fn, uni, hmm_tagset, emit_lp, trans_lp,
            del_idx, score_bi, score_tri, score_phrase, pcfg_parser)


# ═════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════

SAMPLE_PASSAGES = {
    "Passage 1 — General": (
        "The quick brown fox jumps over the lazy dog. "
        "She eats a green salad every afternoon in the quiet courtyard. "
        "I would like to see the world with my friends."
    ),
    "Passage 2 — News": (
        "The company announced a significant increase in international sales today. "
        "Several investors expressed strong confidence in the executive leadership team. "
        "Economic analysts predicted steady growth throughout the upcoming fiscal year."
    ),
    "Passage 3 — Error Demo": (
        "I hav a good feeling about this project. "
        "This is a test sentnce with som simple errors. "
        "Please meat me at the station before the evening train departs."
    ),
}

BADGE_HTML = {
    "SEGMENT-ALERT": "badge-segment",
    "SPELL-ALERT":   "badge-spell",
    "GRAMMAR-ALERT": "badge-grammar",
}

def render_alert(alert):
    cls = BADGE_HTML.get(alert["type"], "badge-grammar")
    st.markdown(f"""
    <div class="alert-card">
        <span class="{cls}">{alert['badge']}</span>
        <div style="margin-top:4px;font-size:0.95rem;">{alert['message']}</div>
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════
# MAIN APP
# ═════════════════════════════════════════════════════════

st.markdown('<div class="main-header">Q4 — Integrated NLP Editor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Word Segmentation (Q1) · Spelling Correction (Q3) · Grammar Checking · PCFG Parsing</div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", [
        "Simulated Fast Typing",
        "Interactive Live Typing",
        "Speed Demon Benchmark",
    ])
    st.markdown("---")
    merge_p = st.slider("Merge Probability (p)", 0.0, 0.25, 0.08, 0.01,
                        help="Prob of dropping space between two words (fast-typing simulation)")
    trigger_n = st.slider("Grammar Trigger Interval (N)", 3, 12, 5,
                          help="Grammar check fires every N processed words")
    grammar_threshold = st.slider("Grammar Alert Threshold (avg log-prob/word)", -15.0, -4.0, -10.0, 0.5,
                                    help="More negative = fewer alerts. -10 is balanced; use -7 for strict checking.")
    st.markdown("---")
    st.caption("**UPOS → PTB tag map:**")
    for k, v in UPOS_TO_PTB.items():
        st.caption(f"`{k}` → `{v}`")

# ── Model loading with step-by-step progress ──────────────
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False

if not st.session_state.models_loaded:
    status = st.status("Loading NLP models (first run only)…", expanded=True)
    with status:
        st.write("Downloading corpora...")
        nltk.download("brown",    quiet=True)
        nltk.download("treebank", quiet=True)

        st.write("Training Q1 Trigram LM + Viterbi…")
        sents = nltk.corpus.brown.tagged_sents(tagset="universal")[:20000]
        uni, bi, tri = train_trigram_lm(sents)
        vocab        = set(uni.keys()) - {"<S>","</S>"}
        log_prob_fn  = make_log_prob(uni, bi, tri, len(vocab))

        st.write("Training Q1 HMM POS Tagger…")
        hmm_emit, hmm_trans, hmm_tag_cnt, hmm_tagset = train_hmm(sents)
        emit_lp, trans_lp = make_hmm_fns(hmm_emit, hmm_trans, hmm_tag_cnt, hmm_tagset)

        st.write("Building Q3 Spelling Index…")
        del_idx = build_delete_index(vocab)

        st.write("Training Q4 N-gram Sentence Scorers…")
        score_bi, score_tri, score_phrase = train_sentence_lms(sents)

        st.write("Inducing PCFG from Penn Treebank (500 trees)…")
        pcfg_parser = train_pcfg(max_trees=500)

        st.session_state.vocab        = vocab
        st.session_state.log_prob_fn  = log_prob_fn
        st.session_state.uni          = uni
        st.session_state.hmm_tagset   = hmm_tagset
        st.session_state.emit_lp      = emit_lp
        st.session_state.trans_lp     = trans_lp
        st.session_state.del_idx      = del_idx
        st.session_state.score_bi     = score_bi
        st.session_state.score_tri    = score_tri
        st.session_state.score_phrase = score_phrase
        st.session_state.pcfg_parser  = pcfg_parser
        st.session_state.sents        = sents
        st.session_state.models_loaded = True
        status.update(label="All models loaded!", state="complete", expanded=False)

# Retrieve from session
vocab        = st.session_state.vocab
log_prob_fn  = st.session_state.log_prob_fn
uni_counts   = st.session_state.uni
hmm_tagset   = st.session_state.hmm_tagset
emit_lp      = st.session_state.emit_lp
trans_lp     = st.session_state.trans_lp
del_idx      = st.session_state.del_idx
score_bi     = st.session_state.score_bi
score_tri    = st.session_state.score_tri
score_phrase = st.session_state.score_phrase
pcfg_parser  = st.session_state.pcfg_parser

st.markdown("---")

# ═════════════════════════════════════════════════════════
# MODE 1 — SIMULATED FAST TYPING
# ═════════════════════════════════════════════════════════
if mode == "Simulated Fast Typing":
    col_left, col_right = st.columns([1.15, 0.85])

    with col_left:
        st.subheader("Input Passage")
        sel = st.selectbox("Choose a sample:", list(SAMPLE_PASSAGES.keys()))
        input_text = st.text_area("Passage Text:", value=SAMPLE_PASSAGES[sel], height=130)
        col_btn, col_delay = st.columns([1,1])
        with col_btn:
            start = st.button("Start Simulation", type="primary", use_container_width=True)
        with col_delay:
            delay = st.slider("Delay (s/token):", 0.0, 0.5, 0.07, 0.01)

    with col_right:
        st.subheader("Live Alert Feed")
        alert_box = st.container(height=350)

    if start and input_text.strip():
        stream = simulate_fast_typing_merges(input_text, p=merge_p, seed=42)

        st.markdown("**Simulated Typing Stream:**")
        stream_display = st.empty()
        displayed = []

        accumulated    = []
        tokens_proc    = 0
        seg_count      = 0
        spell_count    = 0
        total_seg_lat  = 0.0
        total_spell_lat= 0.0
        all_alerts     = []

        for token in stream:
            displayed.append(token)
            stream_display.markdown(
                f"<div style='padding:10px;background:#F3F4F6;border-radius:8px;"
                f"font-family:monospace;font-size:1.05rem'>{' '.join(displayed)}</div>",
                unsafe_allow_html=True
            )

            alerts, accumulated, seg_count, spell_count, tokens_proc, sl, sp = \
                process_token_with_alerts(
                    token, vocab, log_prob_fn, del_idx, uni_counts,
                    accumulated, score_phrase, trigger_n,
                    grammar_threshold, seg_count, spell_count, tokens_proc
                )
            total_seg_lat   += sl
            total_spell_lat += sp
            all_alerts.extend(alerts)

            for a in alerts:
                with alert_box: render_alert(a)

            if delay > 0: time.sleep(delay)

        st.success("Simulation complete!")

        # Stats
        m1,m2,m3,m4 = st.columns(4)
        n = max(tokens_proc, 1)
        m1.metric("Avg Token Latency", f"{(total_seg_lat+total_spell_lat)/n:.2f} ms/tok")
        m2.metric("Merges Resolved",   seg_count)
        m3.metric("Spelling Fixed",    spell_count)
        m4.metric("Total Alerts",      len(all_alerts))

        # Final passage analysis
        st.markdown("---")
        st.subheader("End-of-Passage Analysis (PCFG vs N-gram)")
        with st.spinner("Tagging and parsing…"):
            pos_tags = viterbi_pos(accumulated, hmm_tagset, emit_lp, trans_lp)
            results  = analyze_final_passage(accumulated, pos_tags, pcfg_parser, score_bi, score_tri)

        df = pd.DataFrame([{
            "#":             r["sentence_idx"],
            "Sentence":      r["sentence_text"],
            "PCFG Prob":     r["pcfg_result"],
            "Bigram Score":  r["bigram_score"],
            "Trigram Score": r["trigram_score"],
            "Chosen Method": r["chosen_method"],
            "Verdict":       r["final_verdict"],
        } for r in results])
        st.dataframe(df, use_container_width=True)

        with st.expander("View PCFG Parse Trees"):
            for r in results:
                st.markdown(f"**Sent {r['sentence_idx']}:** {r['sentence_text']}")
                if r["pcfg_tree"]:
                    st.code(str(r["pcfg_tree"]), language="text")
                else:
                    st.caption("*(Unparseable)*")

# ═════════════════════════════════════════════════════════
# MODE 2 — INTERACTIVE LIVE TYPING
# ═════════════════════════════════════════════════════════
elif mode == "Interactive Live Typing":
    st.subheader("Real-Time Interactive Editor")
    st.caption("Type text below. On every re-run the engine processes tokens incrementally.")

    user_text = st.text_area(
        "Live Input:", height=120,
        placeholder="Try: I hav a good feeling about thissproject..."
    )
    run_live = st.button("Analyse", type="primary")

    if run_live and user_text.strip():
        tokens = user_text.strip().split()
        accumulated  = []
        tokens_proc  = 0
        seg_count    = 0
        spell_count  = 0
        all_alerts   = []

        for token in tokens:
            alerts, accumulated, seg_count, spell_count, tokens_proc, _, _ = \
                process_token_with_alerts(
                    token, vocab, log_prob_fn, del_idx, uni_counts,
                    accumulated, score_phrase, trigger_n,
                    grammar_threshold, seg_count, spell_count, tokens_proc
                )
            all_alerts.extend(alerts)

        col_text, col_alerts = st.columns([1.2, 0.8])
        with col_text:
            st.markdown("**Corrected Output:**")
            st.info(" ".join(accumulated))
            st.caption(f"Merges fixed: {seg_count} | Spelling fixed: {spell_count}")

        with col_alerts:
            st.markdown(f"**Alerts ({len(all_alerts)}):**")
            if all_alerts:
                for a in all_alerts: render_alert(a)
            else:
                st.success("No issues detected!")

        # Final analysis
        st.markdown("---")
        st.subheader("Final Passage Analysis")
        with st.spinner("Tagging and parsing…"):
            pos_tags = viterbi_pos(accumulated, hmm_tagset, emit_lp, trans_lp)
            results  = analyze_final_passage(accumulated, pos_tags, pcfg_parser, score_bi, score_tri)

        for r in results:
            st.markdown(
                f"**Sent {r['sentence_idx']}**: {r['sentence_text']}  \n"
                f"PCFG: `{r['pcfg_result']}` | Bigram: `{r['bigram_score']}` | "
                f"Trigram: `{r['trigram_score']}` → **{r['final_verdict']}**"
            )

# ═════════════════════════════════════════════════════════
# MODE 3 — SPEED DEMON BENCHMARK
# ═════════════════════════════════════════════════════════
elif mode == "Speed Demon Benchmark":
    st.subheader("Part 5: Speed Demon Benchmark — 1,000 Words")
    st.markdown(
        "Benchmarks the latency of the **per-token layer** (Segmentation + Spelling) "
        "vs. the **grammar-trigger layer** run in isolation on exactly 1,000 simulated words."
    )

    if st.button("Run Benchmark", type="primary"):
        random.seed(42)
        vocab_list     = [w for w in vocab if len(w) >= 4]
        batch_words    = random.sample(vocab_list, min(len(vocab_list), 1000))

        # Corrupt 30% with a deleted character
        corrupted = []
        for idx, w in enumerate(batch_words):
            if idx % 3 == 0 and len(w) >= 3:
                corrupted.append(w[:-1])   # typo: delete last char
            else:
                corrupted.append(w)

        prog = st.progress(0, text="Running full pipeline…")

        # 1. Full pipeline (seg + spell + grammar)
        accumulated, tokens_proc = [], 0
        seg_cnt = spell_cnt = 0
        t0 = time.perf_counter()
        for j, token in enumerate(corrupted):
            _, accumulated, seg_cnt, spell_cnt, tokens_proc, _, _ = \
                process_token_with_alerts(
                    token, vocab, log_prob_fn, del_idx, uni_counts,
                    accumulated, score_phrase, trigger_n,
                    grammar_threshold, seg_cnt, spell_cnt, tokens_proc
                )
            if j % 100 == 0:
                prog.progress(j/1000, text=f"Full pipeline: {j}/1000 words…")
        total_pipeline = time.perf_counter() - t0

        prog.progress(1.0, text="Running grammar-only…")

        # 2. Grammar-only in isolation
        t0 = time.perf_counter()
        for i in range(0, len(corrupted), trigger_n):
            window = corrupted[i:i+trigger_n]
            score_phrase(window)
        total_grammar = time.perf_counter() - t0

        prog.empty()

        # Results
        c1, c2, c3 = st.columns(3)
        c1.metric("Full Pipeline (1000 words)", f"{total_pipeline*1000:.1f} ms",
                  f"{total_pipeline:.3f} ms/word avg")
        c2.metric("Grammar-Only (1000 words)",  f"{total_grammar*1000:.1f} ms",
                  f"{total_grammar:.3f} ms/word avg")
        overhead = total_pipeline - total_grammar
        c3.metric("Seg+Spell Overhead",          f"{overhead*1000:.1f} ms",
                  f"{overhead/max(total_pipeline,1e-9)*100:.1f}% of total")

        words_per_sec = 1000 / max(total_pipeline, 1e-9)
        human_ms_per_word = 750  # ~80 wpm

        st.success(
            f"**Conclusion:** The full pipeline processes **{words_per_sec:,.0f} words/sec** "
            f"({total_pipeline/1*1000/1000:.3f} ms/word). "
            f"Human typing speed is ~1 word every {human_ms_per_word} ms. "
            f"The pipeline is **{human_ms_per_word/(total_pipeline/1000*1000 + 1e-9):.0f}× faster** "
            f"than real typing — comfortably live with no throttling needed."
        )
