"""Integrated Background Editor Streamlit Application (Question 4 Part 5).

Features:
- Live typing mode & fast typing streaming simulation (with space-merge probability p).
- Real-time SEGMENT-ALERT, SPELL-ALERT, and GRAMMAR-ALERT feed.
- End-of-passage PCFG constituency parsing and N-gram scoring table.
- Interactive Speed Demon benchmark runner.
"""

import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.q1_segmentation_tagging.pos_tagging import train_pos_tagger
from src.q1_segmentation_tagging.segmentation import train_trigram_lm
from src.q3_spelling_corrector.candidate_gen import preprocess_symmetric_delete
from src.q3_spelling_corrector.spell_check import (
    build_bigram_model,
    build_vocabulary_and_unigram,
)
from src.q4_integrated_editor.passage_analysis import analyze_final_passage
from src.q4_integrated_editor.pcfg_parser import train_pcfg
from src.q4_integrated_editor.typing_simulation import (
    LiveEditorEngine,
    simulate_fast_typing_merges,
)


@st.cache_resource(show_spinner="Training and loading NLP models (Q1, Q3, Q4)...")
def load_all_models():
    """Initializes and caches all models across Q1, Q3, and Q4."""
    import nltk
    try:
        from nltk.corpus import brown
        tagged_sents = brown.tagged_sents()[:20000]
    except (LookupError, AttributeError):
        nltk.download("brown", quiet=True)
        from nltk.corpus import brown
        tagged_sents = brown.tagged_sents()[:20000]

    train_words = [[w for w, _ in s] for s in tagged_sents]

    # Q1 Models
    q1_lm = train_trigram_lm(train_words)
    q1_tagger = train_pos_tagger(tagged_sents)

    # Q3 Models
    q3_vocab, q3_unigram_probs, _ = build_vocabulary_and_unigram(train_words)
    q3_sym_del = preprocess_symmetric_delete(q3_vocab)
    q3_bigram = build_bigram_model(train_words)

    # Q4 PCFG
    pcfg = train_pcfg(max_trees=350)

    return {
        "q1_lm": q1_lm,
        "q1_tagger": q1_tagger,
        "q3_vocab": q3_vocab,
        "q3_unigram_probs": q3_unigram_probs,
        "q3_sym_del": q3_sym_del,
        "q3_bigram": q3_bigram,
        "pcfg": pcfg,
    }


def get_sample_passages() -> Dict[str, str]:
    """Returns curated test passages."""
    return {
        "Passage 1 (General Literature)": (
            "The quick brown fox jumps over the lazy dog. "
            "She eats a green salad every afternoon in the quiet courtyard. "
            "I would like to see the world with my friends. "
            "The committee members arrived early to discuss the urgent proposal."
        ),
        "Passage 2 (News Report)": (
            "The company announced a significant increase in international sales today. "
            "Several investors expressed strong confidence in the executive leadership team. "
            "Government officials met at the central station to sign the trade agreement. "
            "Economic analysts predicted steady growth throughout the upcoming fiscal year."
        ),
        "Passage 3 (Fast-Typing Error Demonstration)": (
            "I hav a good feeling about this project. "
            "This is a test sentnce with som simple errors. "
            "Please meat me at the station before the evening train departs. "
            "The cat sat on the mat while the children played outside."
        ),
    }


def main():
    st.set_page_config(
        page_title="Integrated Background Editor",
        page_icon="✍️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject Custom CSS for premium styling
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #6B7280;
            margin-bottom: 1.5rem;
        }
        .badge-segment {
            background-color: #FEF3C7;
            color: #92400E;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.8rem;
            border: 1px solid #FCD34D;
        }
        .badge-spell {
            background-color: #FEE2E2;
            color: #991B1B;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.8rem;
            border: 1px solid #FCA5A5;
        }
        .badge-grammar {
            background-color: #EDE9FE;
            color: #5B21B6;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.8rem;
            border: 1px solid #C4B5FD;
        }
        .alert-card {
            background: #FFFFFF;
            border-left: 4px solid #3B82F6;
            padding: 10px 14px;
            margin-bottom: 8px;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-header">Integrated Background Editor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Live Word Segmentation (Q1) • Fast Spelling Correction (Q3) • Constituency PCFG Grammar Checking (Q4)</div>',
        unsafe_allow_html=True,
    )

    models = load_all_models()

    # Sidebar Parameters
    st.sidebar.header("⚙️ Editor Settings")
    mode = st.sidebar.radio(
        "Select Operation Mode:",
        ["Simulated Fast Typing", "Interactive Live Typing", "Speed Demon Benchmark"],
    )

    merge_p = st.sidebar.slider(
        "Fast-typing Merge Probability (p):",
        min_value=0.0,
        max_value=0.25,
        value=0.08,
        step=0.01,
        help="Probability of dropping the spacebar between two consecutive words, mimicking fast typist errors.",
    )

    trigger_n = st.sidebar.slider(
        "Grammar Trigger Interval (N words):",
        min_value=3,
        max_value=12,
        value=5,
        step=1,
        help="Number of accumulated words before triggering the contextual bigram and perplexity checks.",
    )

    # -------------------------------------------------------------
    # MODE 1: SIMULATED FAST TYPING
    # -------------------------------------------------------------
    if mode == "Simulated Fast Typing":
        col1, col2 = st.columns([1.1, 0.9])

        with col1:
            st.subheader("📝 Input Passage")
            sample_options = get_sample_passages()
            selected_sample = st.selectbox("Choose a sample passage or enter your own:", list(sample_options.keys()))
            input_text = st.text_area("Passage Text:", value=sample_options[selected_sample], height=130)

            col_btn, col_spd = st.columns([1, 1])
            with col_btn:
                start_btn = st.button("🚀 Start Typing Simulation", type="primary", use_container_width=True)
            with col_spd:
                typing_delay = st.slider("Typing Delay (seconds/token):", 0.0, 0.3, 0.05, 0.01)

        with col2:
            st.subheader("🔔 Live Alert Feed")
            alerts_container = st.container(height=380)

        # Simulation Run
        if start_btn and input_text.strip():
            engine = LiveEditorEngine(
                q1_lm=models["q1_lm"],
                q1_tagger=models["q1_tagger"],
                q3_vocab=models["q3_vocab"],
                q3_unigram_probs=models["q3_unigram_probs"],
                q3_sym_del_dict=models["q3_sym_del"],
                q3_bigram_model=models["q3_bigram"],
                trigger_n=trigger_n,
            )

            # Generate merged stream
            stream_tokens = simulate_fast_typing_merges(input_text, p=merge_p, seed=42)

            st.write("---")
            st.subheader("⌨️ Simulated Incoming Stream")
            stream_display = st.empty()

            displayed_stream: List[str] = []

            for token in stream_tokens:
                displayed_stream.append(token)
                stream_display.markdown(
                    f"<div style='padding: 12px; background: #F3F4F6; border-radius: 8px; font-family: monospace; font-size: 1.05rem;'>{' '.join(displayed_stream)}</div>",
                    unsafe_allow_html=True,
                )

                new_alerts = engine.process_incoming_token(token)
                for alert in new_alerts:
                    badge_class = (
                        "badge-segment"
                        if alert["type"] == "SEGMENT-ALERT"
                        else ("badge-spell" if alert["type"] == "SPELL-ALERT" else "badge-grammar")
                    )
                    with alerts_container:
                        st.markdown(
                            f"""
                            <div class="alert-card">
                                <span class="{badge_class}">{alert['badge']}</span>
                                <div style="margin-top: 4px; font-size: 0.95rem;">{alert['message']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                if typing_delay > 0:
                    time.sleep(typing_delay)

            st.success("✅ Typing simulation completed!")

            # Latency and Statistics
            stats = engine.get_latency_stats()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Per-Token Latency", f"{stats['avg_token_latency_ms']:.2f} ms")
            m2.metric("Grammar Check Latency", f"{stats['avg_grammar_latency_ms']:.2f} ms")
            m3.metric("Merges Resolved", f"{stats['merges_resolved']}")
            m4.metric("Total Alerts Fired", f"{stats['alerts_count']}")

            # Final Passage Analysis
            st.write("---")
            st.subheader("📊 Part 4: End-of-Passage Comparative Analysis")
            with st.spinner("Parsing sentences with PCFG Constituency and N-Gram models..."):
                analysis_rows = analyze_final_passage(
                    engine.accumulated_tokens,
                    models["pcfg"],
                    models["q3_bigram"],
                    models["q1_lm"],
                    models["q1_tagger"],
                    alerts=engine.alerts,
                )

            df_summary = pd.DataFrame(
                [
                    {
                        "Sentence #": r["sentence_idx"],
                        "Sentence Text": r["sentence_text"],
                        "PCFG Score": r["pcfg_result"],
                        "Bigram Log-P": r["bigram_score"],
                        "Trigram Log-P": r["trigram_score"],
                        "Chosen Method": r["chosen_method"],
                        "Final Verdict": r["final_verdict"],
                        "Merges Resolved": r["merges_resolved"],
                        "Spelling Fixed": r["spelling_corrections"],
                    }
                    for r in analysis_rows
                ]
            )

            st.dataframe(df_summary, use_container_width=True)

            # PCFG Tree Display
            with st.expander("🌳 View PCFG Constituency Parse Trees"):
                for r in analysis_rows:
                    st.markdown(f"**Sentence {r['sentence_idx']}:** {r['sentence_text']}")
                    if r["pcfg_tree"] is not None:
                        st.code(str(r["pcfg_tree"]), language="text")
                    else:
                        st.caption("*(Unparseable under current PCFG grammar)*")

    # -------------------------------------------------------------
    # MODE 2: INTERACTIVE LIVE TYPING
    # -------------------------------------------------------------
    elif mode == "Interactive Live Typing":
        st.subheader("⌨️ Real-Time Interactive Editor")
        st.write("Type or paste any text below. The background engine will process it incrementally.")

        user_text = st.text_area(
            "Live Text Input:",
            value="I hav a good feeling about this test sentnce.",
            height=120,
        )

        if user_text:
            engine = LiveEditorEngine(
                q1_lm=models["q1_lm"],
                q1_tagger=models["q1_tagger"],
                q3_vocab=models["q3_vocab"],
                q3_unigram_probs=models["q3_unigram_probs"],
                q3_sym_del_dict=models["q3_sym_del"],
                q3_bigram_model=models["q3_bigram"],
                trigger_n=trigger_n,
            )

            tokens = user_text.split()
            for t in tokens:
                engine.process_incoming_token(t)

            col_text, col_alerts = st.columns([1.2, 0.8])
            with col_text:
                st.markdown("**Corrected Text Output:**")
                st.info(" ".join(engine.accumulated_tokens))

            with col_alerts:
                st.markdown(f"**Active Alerts ({len(engine.alerts)}):**")
                for alert in engine.alerts:
                    badge_class = (
                        "badge-segment"
                        if alert["type"] == "SEGMENT-ALERT"
                        else ("badge-spell" if alert["type"] == "SPELL-ALERT" else "badge-grammar")
                    )
                    st.markdown(
                        f"""
                        <div class="alert-card">
                            <span class="{badge_class}">{alert['badge']}</span>
                            <div style="margin-top: 4px; font-size: 0.95rem;">{alert['message']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            stats = engine.get_latency_stats()
            st.caption(
                f"⚡ Avg token latency: **{stats['avg_token_latency_ms']:.2f} ms** | Avg grammar latency: **{stats['avg_grammar_latency_ms']:.2f} ms**"
            )

    # -------------------------------------------------------------
    # MODE 3: SPEED DEMON BENCHMARK
    # -------------------------------------------------------------
    elif mode == "Speed Demon Benchmark":
        st.subheader("⚡ Part 5: Speed Demon Benchmark (1,000 Words)")
        st.write(
            "Benchmarks the latency difference between the per-token layer (Segmentation + Spelling) "
            "and the grammar-trigger layer on exactly 1,000 simulated words."
        )

        if st.button("Run 1,000-Word Speed Demon Benchmark", type="primary"):
            with st.spinner("Generating batch and measuring latency..."):
                random.seed(42)
                vocab_list = [w for w in models["q3_vocab"] if len(w) >= 4]
                batch_words = random.sample(vocab_list, min(len(vocab_list), 1000))

                # Introduce typos into 30% of words
                corrupted_batch = []
                for idx, w in enumerate(batch_words):
                    if idx % 3 == 0 and len(w) >= 3:
                        # typo: delete a character
                        corrupted_batch.append(w[:-1])
                    else:
                        corrupted_batch.append(w)

                engine = LiveEditorEngine(
                    q1_lm=models["q1_lm"],
                    q1_tagger=models["q1_tagger"],
                    q3_vocab=models["q3_vocab"],
                    q3_unigram_probs=models["q3_unigram_probs"],
                    q3_sym_del_dict=models["q3_sym_del"],
                    q3_bigram_model=models["q3_bigram"],
                    trigger_n=trigger_n,
                )

                # 1. Full pipeline
                t0 = time.perf_counter()
                for token in corrupted_batch:
                    engine.process_incoming_token(token)
                total_pipeline_time = time.perf_counter() - t0

                # 2. Grammar only in isolation
                t0 = time.perf_counter()
                for i in range(0, len(corrupted_batch), trigger_n):
                    window = corrupted_batch[i : i + trigger_n]
                    models["q3_bigram"].score_phrase(window)
                total_grammar_time = time.perf_counter() - t0

            c1, c2, c3 = st.columns(3)
            c1.metric("Full Pipeline Time", f"{total_pipeline_time:.4f} s", f"{total_pipeline_time/1000*1000:.3f} ms/word")
            c2.metric("Grammar-Only Time", f"{total_grammar_time:.4f} s", f"{total_grammar_time/1000*1000:.3f} ms/word")
            overhead = total_pipeline_time - total_grammar_time
            c3.metric("Segmentation+Spelling Overhead", f"{overhead:.4f} s", f"{overhead/total_pipeline_time*100:.1f}%")

            st.success(
                f"**Benchmark Conclusion**: The per-token segmentation+spelling layer processes at "
                f"**{total_pipeline_time/1000*1000:.3f} ms/word** (~{int(1000/total_pipeline_time):,} words/sec), "
                f"which is far faster than human typing speed (typically ~100-200 ms per keystroke). "
                f"It is cheap enough to run live on every keystroke without lag."
            )


if __name__ == "__main__":
    main()
