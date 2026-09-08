"""
Question 4 — Integrated NLP Editor (Streamlit App)
Self-contained: all helper functions integrated.
Run from project root:
    uv run streamlit run src/streamlit_app.py
"""

import math
import random
import time
from collections import Counter, defaultdict
from typing import Optional, Tuple

import nltk
from nltk import Nonterminal, Tree
import pandas as pd
import streamlit as st

from src.pcfg_parser import train_pcfg, parse_with_pcfg, UPOS_TO_PTB
from src.q1_funcs import (
    train_trigram_lm,
    make_log_prob,
    viterbi_segment,
    extract_morphological_features,
    MorphologyFeatureModel,
    train_hmm,
    make_hmm_fns,
    beam_pos_tagger,
    viterbi_pos,
    joint_beam_segment_token,
)
from src.q3_funcs import (
    damerau_levenshtein,
    _generate_edit1_strings,
    method_a_candidates,
    build_delete_index,
    method_b_candidates,
    best_unigram_candidate,
    best_candidate_with_context,
    bigram_probability,
    real_word_context_score,
    correct_real_word,
)

# ─────────────────────────────────────────────────────────
# Page config — must be the very first Streamlit call
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Q4 - Integrated NLP Editor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# Custom CSS (No emojis)
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
.badge-realword {
    background:#FCE7F3; color:#9D174D; padding:2px 8px;
    border-radius:4px; font-weight:600; font-size:0.8rem; border:1px solid #FBCFE8;
}
.alert-card {
    background:#FFFFFF; border-left:4px solid #3B82F6;
    padding:10px 14px; margin-bottom:8px; border-radius:6px;
    box-shadow:0 1px 3px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════
# SECTION A — Passage Sampler (Q4 Feature)
# ═════════════════════════════════════════════════════════

def sample_random_passage(corpus_name="gutenberg", num_sentences=6, seed=None):
    """Samples a contiguous passage of 5-8 sentences from gutenberg, brown, or reuters."""
    if seed is not None:
        random.seed(seed)
    try:
        if corpus_name == "gutenberg":
            nltk.download("gutenberg", quiet=True)
            fids = nltk.corpus.gutenberg.fileids()
            fid = random.choice(fids)
            sents = nltk.corpus.gutenberg.sents(fid)
        elif corpus_name == "brown":
            nltk.download("brown", quiet=True)
            fids = nltk.corpus.brown.fileids()
            fid = random.choice(fids)
            sents = nltk.corpus.brown.sents(fid)
        elif corpus_name == "reuters":
            nltk.download("reuters", quiet=True)
            fids = nltk.corpus.reuters.fileids()
            fid = random.choice(fids)
            sents = nltk.corpus.reuters.sents(fid)
        else:
            return ""

        if len(sents) <= num_sentences:
            chosen = sents
        else:
            max_idx = len(sents) - num_sentences
            start = random.randint(0, max_idx)
            chosen = sents[start : start + num_sentences]

        text = " ".join(" ".join(s) for s in chosen)
        for p in [".", ",", "!", "?", ";", ":", "'", '"']:
            text = text.replace(f" {p}", p)
        return text
    except Exception:
        return (
            "The quick brown fox jumps over the lazy dog. "
            "She eats a green salad every afternoon in the quiet courtyard. "
            "I would like to see the world with my friends."
        )

# ═════════════════════════════════════════════════════════
# SECTION B — Q4: Tagset Reconciliation + PCFG + N-gram
# ═════════════════════════════════════════════════════════

def train_sentence_lms(sents, k=0.01):
    """Trains Bigram and Trigram LMs on whole sentences for passage scoring."""
    uni, bi, tri = Counter(), Counter(), Counter()
    bi_by_prev = defaultdict(Counter)
    for sent in sents:
        words = ["<s>", "<s>"] + [w.lower() for w, _ in sent] + ["</s>"]
        for w in words:
            uni[w] += 1
        for i in range(len(words) - 1):
            bi[(words[i], words[i+1])] += 1
            bi_by_prev[words[i]][words[i+1]] += 1
        for i in range(len(words) - 2):
            tri[(words[i], words[i+1], words[i+2])] += 1
    v_size = len(uni)

    def score_bigram(words):
        w = ["<s>"] + [x.lower() for x in words] + ["</s>"]
        return sum(math.log((bi.get((w[i], w[i+1]), 0) + k) / (uni.get(w[i], 0) + k * v_size)) for i in range(len(w) - 1))

    def score_trigram(words):
        w = ["<s>", "<s>"] + [x.lower() for x in words] + ["</s>"]
        return sum(math.log((tri.get((w[i], w[i+1], w[i+2]), 0) + k) / (bi.get((w[i], w[i+1]), 0) + k * v_size)) for i in range(len(w) - 2))

    def score_phrase_bigram(words):
        if len(words) < 2:
            return 0.0
        return score_bigram(words) / len(words)

    return score_bigram, score_trigram, score_phrase_bigram, bi_by_prev, uni

# ═════════════════════════════════════════════════════════
# SECTION C — Typing Simulation Engine
# ═════════════════════════════════════════════════════════

def simulate_fast_typing_merges(text, p=0.08, seed=None):
    """Randomly drop spaces between words with probability p."""
    if seed is not None:
        random.seed(seed)
    raw = text.strip().split()
    if not raw:
        return []
    merged, i = [], 0
    while i < len(raw):
        if i < len(raw) - 1 and random.random() < p:
            merged.append(raw[i] + raw[i+1])
            i += 2
        else:
            merged.append(raw[i])
            i += 1
    return merged


def process_token_with_alerts(
    token, vocab, log_prob_fn, delete_index, uni_counts,
    accumulated, score_phrase_bigram, trigger_n,
    grammar_threshold, seg_count, spell_count, tokens_processed,
    hmm_tagset=None, emit_lp=None, trans_lp=None,
    bi_by_prev=None, uni_lm=None, candidate_method="B"
):
    """Process one token through Segment -> Spell -> Grammar checks."""
    alerts = []
    clean = "".join(c for c in token if c.isalnum()).lower()
    punct = "".join(c for c in token if not c.isalnum())

    # ── 1. SEGMENT-ALERT ──────────────────────────────────
    t0_seg = time.perf_counter()
    active = []
    if clean and (clean not in vocab or len(clean) >= 12):
        if hmm_tagset is not None and emit_lp is not None and trans_lp is not None:
            splits, split_tags, did_split = joint_beam_segment_token(
                clean, vocab, log_prob_fn, hmm_tagset, emit_lp, trans_lp,
                alpha=1.0, beta=0.5, beam_width=10
            )
        else:
            splits = viterbi_segment(clean, vocab, log_prob_fn)
            split_tags = ["NOUN"] * len(splits)
            did_split = (len(splits) >= 2 and all((len(w) >= 2 or w in {"a", "i"}) and w in vocab for w in splits))

        if did_split:
            tagged_str = " | ".join(f"{w} [{t}]" for w, t in zip(splits, split_tags))
            alerts.append({
                "type": "SEGMENT-ALERT",
                "badge": "SEGMENT",
                "original": token,
                "message": f"Merged token '{token}' -> {tagged_str}"
            })
            seg_count += 1
            active.extend(splits)
        else:
            active.append(clean)
    else:
        active.append(clean)
    seg_lat = (time.perf_counter() - t0_seg) * 1000.0

    # ── 2. SPELL-ALERT ────────────────────────────────────
    t0_spell = time.perf_counter()
    spell_checked = []
    for t in active:
        t_clean = t.lower()
        if t_clean and t_clean not in vocab:
            if candidate_method == "A":
                cands = method_a_candidates(t_clean, vocab)
            else:
                cands = method_b_candidates(t_clean, delete_index)
            prev_tok = accumulated[-1] if (accumulated and accumulated[-1].isalpha()) else None
            best = best_candidate_with_context(t_clean, cands, uni_counts, prev_word=prev_tok, bi_by_prev=bi_by_prev)
            if best and best != t_clean:
                alerts.append({
                    "type": "SPELL-ALERT",
                    "badge": "SPELL",
                    "original": t,
                    "message": f"Non-word '{t}' -> corrected to '{best}'"
                })
                spell_count += 1
                spell_checked.append(best)
            else:
                spell_checked.append(t_clean)
        else:
            spell_checked.append(t_clean)
    spell_lat = (time.perf_counter() - t0_spell) * 1000.0

    # ── 3. GRAMMAR-ALERT & Real-Word Check every N tokens ─
    gram_lat = 0.0
    STOP_WORDS = {
        "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
        "which", "this", "that", "these", "those", "then", "just", "so", "than",
        "such", "both", "through", "about", "for", "is", "of", "while", "during",
        "to", "from", "in", "out", "on", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how", "all",
        "any", "both", "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "can", "will", "just", "don", "should", "now", "i", "me", "my", "myself",
        "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
        "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
        "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "having", "do", "does", "did", "doing", "would", "should", "could",
        "ought"
    }

    for tok in spell_checked:
        accumulated.append(tok)
        tokens_processed += 1

        if tokens_processed % trigger_n == 0:
            t0_gram = time.perf_counter()
            window_size = min(len(accumulated), trigger_n + 2)
            window = [w for w in accumulated[-window_size:] if w.isalpha()]

            if len(window) >= 3:
                avg_log_p = score_phrase_bigram(window)
                if avg_log_p < grammar_threshold:
                    ppl = math.exp(-avg_log_p) if avg_log_p > -20 else 9999.0

                    # Contextual Real-Word Error Detection:
                    # Within the suspicious phrase, check if a content word is an in-vocab error
                    real_word_alerted = False
                    if bi_by_prev is not None and uni_lm is not None:
                        start_w = max(0, len(accumulated) - window_size)
                        for w_i in range(start_w, len(accumulated)):
                            w_curr = accumulated[w_i]
                            w_lower = w_curr.lower()
                            if w_lower in STOP_WORDS or len(w_lower) < 3 or not w_curr.isalpha():
                                continue
                            prev_w = accumulated[w_i - 1].lower() if w_i > 0 and accumulated[w_i - 1].isalpha() else "<s>"
                            next_w = accumulated[w_i + 1].lower() if w_i + 1 < len(accumulated) and accumulated[w_i + 1].isalpha() else "</s>"
                            corr = correct_real_word(
                                w_lower, prev_w, next_w, delete_index,
                                bi_by_prev, uni_lm, len(vocab),
                                improvement_threshold=3.0, method=candidate_method, vocab=vocab
                            )
                            if corr != w_lower and corr not in STOP_WORDS:
                                alerts.append({
                                    "type": "GRAMMAR-ALERT",
                                    "badge": "REAL-WORD",
                                    "original": w_curr,
                                    "message": f"Real-word contextual error in phrase: '{w_curr}' -> corrected to '{corr}'"
                                })
                                accumulated[w_i] = corr
                                real_word_alerted = True

                    if not real_word_alerted:
                        alerts.append({
                            "type": "GRAMMAR-ALERT",
                            "badge": "GRAMMAR",
                            "original": " ".join(window),
                            "message": f"Unusual sequence (perplexity={ppl:.1f}): '{' '.join(window)}'"
                        })
            gram_lat += (time.perf_counter() - t0_gram) * 1000.0

    for ch in punct:
        if ch in {".", "!", "?", ";"}:
            accumulated.append(ch)

    return alerts, accumulated, seg_count, spell_count, tokens_processed, seg_lat, spell_lat, gram_lat


def analyze_final_passage(accumulated, pos_tags, pcfg_grammar, score_bi, score_tri, all_alerts=None):
    """Splits passage into sentences, evaluates PCFG vs N-gram, and reconstructs parse trees."""
    rows = []
    current, cur_tags = [], []
    for w, t in zip(accumulated, pos_tags):
        current.append(w)
        cur_tags.append(t)
        if w in {".", "!", "?", ";"}:
            rows.append((current[:], cur_tags[:]))
            current, cur_tags = [], []
    if current:
        rows.append((current, cur_tags))
    if not rows:
        rows = [(accumulated, pos_tags)]

    results = []
    for i, (sent_words, sent_tags) in enumerate(rows):
        tree, log_p, status = parse_with_pcfg(pcfg_grammar, sent_words, sent_tags)
        bi_score  = score_bi(sent_words)
        tri_score = score_tri(sent_words)
        n = max(len(sent_words), 1)
        avg_tri = tri_score / n

        if status == "parsed":
            chosen  = "PCFG (Constituency)"
            verdict = "Grammatical (Valid Structure)"
            pcfg_display = f"{log_p:.2f}"
        elif status == "partial":
            chosen  = "PCFG (Partial)"
            verdict = "Possibly Grammatical (Partial Parse)"
            pcfg_display = f"{log_p:.2f} (partial)"
        elif avg_tri > -9.0:
            chosen  = "Trigram"
            verdict = "Grammatical (Trigram OK)"
            pcfg_display = "Unparseable"
        else:
            chosen  = "Bigram/Trigram"
            verdict = "Possibly Ungrammatical"
            pcfg_display = "Unparseable"

        merges_count = 0
        spells_count = 0
        if all_alerts:
            sent_word_set = {w.lower() for w in sent_words if w.isalnum()}
            import re as _re
            for a in all_alerts:
                orig = a.get("original", "").lower()
                badge = a.get("badge", "")
                # Also extract corrected/split words from message (after "->")
                # e.g. "Non-word 'sentenc' -> corrected to 'sentence'"
                #      "Merged token 'tobe' -> to [ADP] | be [DET]"
                msg = a.get("message", "")
                corrected_words = set()
                if "->" in msg:
                    after = msg.split("->", 1)[1]
                    after_clean = _re.sub(r"\[[A-Z]+\]", "", after)
                    for part in _re.split(r"[\s|,]+", after_clean):
                        part = part.strip().strip("'\"").lower()
                        if part and part.isalpha():
                            corrected_words.add(part)
                # Match if original OR any corrected token appears in the sentence
                all_candidate_words = set(orig.split()) | corrected_words
                if any(w in sent_word_set for w in all_candidate_words):
                    if badge == "SEGMENT":
                        merges_count += 1
                    elif badge in {"SPELL", "REAL-WORD"}:
                        spells_count += 1

        results.append({
            "sentence_idx":  i + 1,
            "sentence_text": " ".join(sent_words),
            "pcfg_result":   pcfg_display,
            "bigram_score":  f"{bi_score:.2f}",
            "trigram_score": f"{tri_score:.2f}",
            "chosen_method": chosen,
            "final_verdict": verdict,
            "merges_resolved": merges_count,
            "spelling_corrections": spells_count,
            "tree": tree,
        })
    return results

# ═════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════

SAMPLE_PASSAGES = {
    "Passage 1 - General": (
        "The quick brown fox jumps over the lazy dog. "
        "She eats a green salad every afternoon in the quiet courtyard. "
        "I would like to see the world with my friends."
    ),
    "Passage 2 - News": (
        "The company announced a significant increase in international sales today. "
        "Several investors expressed strong confidence in the executive leadership team. "
        "Economic analysts predicted steady growth throughout the upcoming fiscal year."
    ),
    "Passage 3 - Error Demo": (
        "I hav a good feeling about thissproject. "
        "This is a test sentnce with som simple errors. "
        "Please meat me at the station before the evening train departs."
    ),
}

BADGE_HTML = {
    "SEGMENT":   "badge-segment",
    "SPELL":     "badge-spell",
    "GRAMMAR":   "badge-grammar",
    "REAL-WORD": "badge-realword",
}

def render_alert(alert):
    badge_key = alert.get("badge", "GRAMMAR")
    cls = BADGE_HTML.get(badge_key, "badge-grammar")
    st.markdown(f"""
    <div class="alert-card">
        <span class="{cls}">{alert['badge']}</span>
        <div style="margin-top:4px;font-size:0.95rem;">{alert['message']}</div>
    </div>""", unsafe_allow_html=True)


def perturb_token_edit1(w):
    """Perturbs a word with a realistic edit-distance-1 deletion, insertion, substitution, or transposition."""
    if len(w) < 2:
        return w
    edit_type = random.choice(["delete", "insert", "substitute", "transpose"])
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    n = len(w)
    if edit_type == "delete" and n > 2:
        i = random.randint(0, n - 1)
        return w[:i] + w[i+1:]
    elif edit_type == "insert":
        i = random.randint(0, n)
        c = random.choice(alphabet)
        return w[:i] + c + w[i:]
    elif edit_type == "substitute":
        i = random.randint(0, n - 1)
        c = random.choice([ch for ch in alphabet if ch != w[i]])
        return w[:i] + c + w[i+1:]
    elif edit_type == "transpose" and n > 2:
        i = random.randint(0, n - 2)
        return w[:i] + w[i+1] + w[i] + w[i+2:]
    return w[:-1]

# ═════════════════════════════════════════════════════════
# MAIN APP
# ═════════════════════════════════════════════════════════

st.markdown('<div class="main-header">Q4 - Integrated NLP Editor</div>', unsafe_allow_html=True)
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
    candidate_method = st.radio(
        "Candidate Generation Method",
        ["Method B (Symmetric Delete)", "Method A (Edit-1 + DL Verification)"],
        index=0,
        help="Choose between symmetric delete indexing (Method B) or edit-1 generation with DL verification (Method A)."
    )
    method_flag = "A" if "Method A" in candidate_method else "B"

    st.markdown("---")
    merge_p = st.slider("Merge Probability (p)", 0.0, 0.25, 0.08, 0.01,
                        help="Prob of dropping space between two words (fast-typing simulation)")
    trigger_n = st.slider("Grammar Trigger Interval (N)", 3, 12, 5,
                          help="Grammar check fires every N processed words")
    grammar_threshold = st.slider("Grammar Alert Threshold (avg log-prob/word)", -15.0, -4.0, -7.5, 0.5)
    st.markdown("---")
    st.caption("**UPOS -> PTB tag map:**")
    for k, v in UPOS_TO_PTB.items():
        st.caption(f"`{k}` -> `{v}`")

# ── Model loading with step-by-step progress ──────────────
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False

if not st.session_state.models_loaded:
    status = st.status("Loading NLP models (first run only)...", expanded=True)
    with status:
        st.write("Downloading corpora...")
        nltk.download("universal_tagset", quiet=True)
        nltk.download("brown", quiet=True)
        nltk.download("treebank", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

        st.write("Training Q1 Trigram LM...")
        sents = nltk.corpus.brown.tagged_sents(tagset="universal")
        uni, bi, tri = train_trigram_lm(sents)
        vocab = set(uni.keys()) - {"<S>", "</S>"}
        log_prob_fn = make_log_prob(uni, bi, tri, len(vocab))

        st.write("Training Q1 Feature-Based POS Classifier + Beam Decoder...")
        hmm_emit, hmm_trans, hmm_tag_cnt, hmm_tagset, hmm_morph = train_hmm(sents)
        emit_lp, trans_lp = make_hmm_fns(hmm_emit, hmm_trans, hmm_tag_cnt, hmm_tagset, hmm_morph)

        st.write("Building Q3 Spelling Delete Index...")
        del_idx = build_delete_index(vocab)

        st.write("Training Q4 N-gram Sentence Scorers...")
        score_bi, score_tri, score_phrase, bi_by_prev, uni_lm = train_sentence_lms(sents)

        st.write("Inducing PCFG from Penn Treebank...")
        pcfg_parser = train_pcfg()

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
        st.session_state.bi_by_prev   = bi_by_prev
        st.session_state.uni_lm       = uni_lm
        st.session_state.pcfg_parser  = pcfg_parser
        st.session_state.sents        = sents
        st.session_state.models_loaded = True
        if hasattr(status, "update"):
            try:
                status.update(label="All models loaded!", state="complete", expanded=False)
            except Exception:
                pass

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
bi_by_prev   = st.session_state.bi_by_prev
uni_lm       = st.session_state.uni_lm
pcfg_parser  = st.session_state.pcfg_parser

st.markdown("---")

# ═════════════════════════════════════════════════════════
# MODE 1 — SIMULATED FAST TYPING
# ═════════════════════════════════════════════════════════
if mode == "Simulated Fast Typing":
    col_left, col_right = st.columns([1.15, 0.85])

    with col_left:
        st.subheader("Input Passage")
        sel = st.selectbox("Choose a sample passage:", list(SAMPLE_PASSAGES.keys()))
        input_text = st.text_area("Passage Text:", value=SAMPLE_PASSAGES[sel], height=130)

        col_btn, col_delay = st.columns([1, 1])
        with col_btn:
            start = st.button("Start Simulation", type="primary", use_container_width=True)
        with col_delay:
            delay = st.slider("Delay (s/token):", 0.0, 0.5, 0.07, 0.01)

    with col_right:
        st.subheader("Live Alert Feed")
        alert_box = st.container(height=380)

    if start and input_text.strip():
        stream = simulate_fast_typing_merges(input_text, p=merge_p, seed=42)

        st.markdown("**Simulated Typing Stream:**")
        stream_display = st.empty()
        displayed = []

        accumulated     = []
        tokens_proc     = 0
        seg_count       = 0
        spell_count     = 0
        total_seg_lat   = 0.0
        total_spell_lat = 0.0
        total_gram_lat  = 0.0
        all_alerts      = []

        for token in stream:
            displayed.append(token)
            stream_display.markdown(
                f"<div style='padding:10px;background:#F3F4F6;border-radius:8px;"
                f"font-family:monospace;font-size:1.05rem'>{' '.join(displayed)}</div>",
                unsafe_allow_html=True
            )

            alerts, accumulated, seg_count, spell_count, tokens_proc, sl, sp, gl = \
                process_token_with_alerts(
                    token, vocab, log_prob_fn, del_idx, uni_counts,
                    accumulated, score_phrase, trigger_n,
                    grammar_threshold, seg_count, spell_count, tokens_proc,
                    hmm_tagset=hmm_tagset, emit_lp=emit_lp, trans_lp=trans_lp,
                    bi_by_prev=bi_by_prev, uni_lm=uni_lm, candidate_method=method_flag
                )
            total_seg_lat   += sl
            total_spell_lat += sp
            total_gram_lat  += gl
            all_alerts.extend(alerts)

            for a in alerts:
                with alert_box:
                    render_alert(a)

            if delay > 0:
                time.sleep(delay)

        st.success("Simulation complete!")

        # Latency & Alert Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        n = max(tokens_proc, 1)
        num_triggers = max(tokens_proc // trigger_n, 1)
        avg_tok_lat = (total_seg_lat + total_spell_lat) / n
        avg_gram_lat = total_gram_lat / num_triggers

        m1.metric("Token Latency (Seg+Spell)", f"{avg_tok_lat:.2f} ms/tok")
        m2.metric("Grammar Latency", f"{avg_gram_lat:.2f} ms/trig")
        m3.metric("Merges Resolved", seg_count)
        m4.metric("Spelling Fixed", spell_count)
        m5.metric("Total Alerts", len(all_alerts))

        # Final passage analysis
        st.markdown("---")
        st.subheader("End-of-Passage Analysis (PCFG vs N-gram)")
        with st.spinner("Tagging and parsing..."):
            pos_tags = beam_pos_tagger(accumulated, hmm_tagset, emit_lp, trans_lp, beam_width=5)
            results  = analyze_final_passage(accumulated, pos_tags, pcfg_parser, score_bi, score_tri, all_alerts=all_alerts)

        df = pd.DataFrame([{
            "#":                    r["sentence_idx"],
            "Sentence":             r["sentence_text"],
            "PCFG Prob":            r["pcfg_result"],
            "Bigram Score":         r["bigram_score"],
            "Trigram Score":        r["trigram_score"],
            "Chosen Method":        r["chosen_method"],
            "Verdict":              r["final_verdict"],
            "Merges Resolved":      r["merges_resolved"],
            "Spelling Corrections": r["spelling_corrections"],
        } for r in results])
        st.dataframe(df, use_container_width=True)

        with st.expander("View PCFG Parse Details"):
            for r in results:
                st.markdown(f"**Sentence {r['sentence_idx']}:** {r['sentence_text']}")
                pcfg_val = r["pcfg_result"]
                if "Unparseable" in pcfg_val:
                    st.caption("(Unparseable - fell back to n-gram scoring)")
                elif "partial" in pcfg_val:
                    st.caption(f"Partial parse (log-prob: {pcfg_val})")
                else:
                    st.success(f"Full PCFG parse | log-prob: {pcfg_val}")

                if r.get("tree"):
                    st.code(r["tree"].pformat())
                else:
                    st.caption("(No parse tree generated)")

# ═════════════════════════════════════════════════════════
# MODE 2 — INTERACTIVE LIVE TYPING
# ═════════════════════════════════════════════════════════
elif mode == "Interactive Live Typing":
    st.subheader("Real-Time Interactive Editor")
    st.caption("Type text below. The engine processes tokens incrementally.")

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
            alerts, accumulated, seg_count, spell_count, tokens_proc, _, _, _ = \
                process_token_with_alerts(
                    token, vocab, log_prob_fn, del_idx, uni_counts,
                    accumulated, score_phrase, trigger_n,
                    grammar_threshold, seg_count, spell_count, tokens_proc,
                    hmm_tagset=hmm_tagset, emit_lp=emit_lp, trans_lp=trans_lp,
                    bi_by_prev=bi_by_prev, uni_lm=uni_lm, candidate_method=method_flag
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
                for a in all_alerts:
                    render_alert(a)
            else:
                st.success("No issues detected!")

        # Final analysis
        st.markdown("---")
        st.subheader("Final Passage Analysis")
        with st.spinner("Tagging and parsing..."):
            pos_tags = beam_pos_tagger(accumulated, hmm_tagset, emit_lp, trans_lp, beam_width=5)
            results  = analyze_final_passage(accumulated, pos_tags, pcfg_parser, score_bi, score_tri, all_alerts=all_alerts)

        df = pd.DataFrame([{
            "#":                    r["sentence_idx"],
            "Sentence":             r["sentence_text"],
            "PCFG Prob":            r["pcfg_result"],
            "Bigram Score":         r["bigram_score"],
            "Trigram Score":        r["trigram_score"],
            "Chosen Method":        r["chosen_method"],
            "Verdict":              r["final_verdict"],
            "Merges Resolved":      r["merges_resolved"],
            "Spelling Corrections": r["spelling_corrections"],
        } for r in results])
        st.dataframe(df, use_container_width=True)

        with st.expander("View PCFG Parse Details"):
            for r in results:
                st.markdown(f"**Sentence {r['sentence_idx']}:** {r['sentence_text']}")
                if r.get("tree"):
                    st.code(r["tree"].pformat())
                else:
                    st.caption("(No parse tree generated)")

# ═════════════════════════════════════════════════════════
# MODE 3 — SPEED DEMON BENCHMARK
# ═════════════════════════════════════════════════════════
elif mode == "Speed Demon Benchmark":
    st.subheader("Part 5: Speed Demon Benchmark — 1,000 Words")
    st.markdown(
        "Benchmarks the latency of the **per-token layer** (Segmentation + Spelling) "
        "vs. the **grammar-trigger layer** (including contextual Real-Word checks) on exactly 1,000 simulated words."
    )

    if st.button("Run Benchmark", type="primary"):
        random.seed(42)
        vocab_list = [w for w in vocab if len(w) >= 4]
        batch_words = random.sample(vocab_list, min(len(vocab_list), 1000))

        # Corrupt 30% with realistic edit-distance-1 errors
        corrupted = []
        for idx, w in enumerate(batch_words):
            if idx % 3 == 0 and len(w) >= 3:
                corrupted.append(perturb_token_edit1(w))
            else:
                corrupted.append(w)

        prog = st.progress(0, text="Running full pipeline...")

        # 1. Full pipeline (seg + spell + grammar)
        accumulated, tokens_proc = [], 0
        seg_cnt = spell_cnt = 0
        t0 = time.perf_counter()
        for j, token in enumerate(corrupted):
            _, accumulated, seg_cnt, spell_cnt, tokens_proc, _, _, _ = \
                process_token_with_alerts(
                    token, vocab, log_prob_fn, del_idx, uni_counts,
                    accumulated, score_phrase, trigger_n,
                    grammar_threshold, seg_cnt, spell_cnt, tokens_proc,
                    hmm_tagset=hmm_tagset, emit_lp=emit_lp, trans_lp=trans_lp,
                    bi_by_prev=bi_by_prev, uni_lm=uni_lm, candidate_method=method_flag
                )
            if j % 100 == 0:
                prog.progress(j / 1000, text=f"Full pipeline: {j}/1000 words...")
        total_pipeline = time.perf_counter() - t0

        prog.progress(1.0, text="Running grammar-only pass...")

        # 2. Grammar-only in isolation (phrase score + contextual real-word checking)
        t0 = time.perf_counter()
        for i in range(0, len(corrupted), trigger_n):
            window = corrupted[i : i + trigger_n]
            score_phrase(window)
            for k in range(len(window)):
                w_curr = window[k]
                prev_w = window[k - 1] if k > 0 else "<s>"
                next_w = window[k + 1] if k + 1 < len(window) else "</s>"
                correct_real_word(
                    w_curr, prev_w, next_w, del_idx,
                    bi_by_prev, uni_lm, len(vocab),
                    improvement_threshold=0.5, method=method_flag, vocab=vocab
                )
        total_grammar = time.perf_counter() - t0

        prog.empty()

        # Results
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Full Pipeline (1000 words)",
            f"{total_pipeline * 1000:.1f} ms",
            f"{total_pipeline:.3f} ms/word avg"
        )
        c2.metric(
            "Grammar-Only (1000 words)",
            f"{total_grammar * 1000:.1f} ms",
            f"{total_grammar:.3f} ms/word avg"
        )
        overhead = total_pipeline - total_grammar
        c3.metric(
            "Seg+Spell Overhead",
            f"{overhead * 1000:.1f} ms",
            f"{overhead / max(total_pipeline, 1e-9) * 100:.1f}% of total"
        )

        words_per_sec = 1000 / max(total_pipeline, 1e-9)
        human_ms_per_word = 750  # ~80 wpm

        st.success(
            f"**Conclusion:** The full pipeline processes **{words_per_sec:,.0f} words/sec** "
            f"({total_pipeline / 1000 * 1000:.3f} ms/word). "
            f"Human typing speed is ~1 word every {human_ms_per_word} ms. "
            f"The pipeline is **{human_ms_per_word / (total_pipeline / 1000 * 1000 + 1e-9):.0f}x faster** "
            f"than real typing - comfortably live with no throttling needed."
        )
