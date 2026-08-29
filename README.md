# NLP Group Assignment 1

This repository contains the implementation for the NLP Group Assignment, focusing on various core Natural Language Processing tasks ranging from word segmentation to building a transition-based dependency parser and an integrated Streamlit-based editor.

## Project Structure

```text
.
├── data/                             # Dataset directory
│   ├── processed/                    # Cleaned/processed datasets
│   └── raw/                          # Raw corpora (Brown, UD treebanks, CoNLL-U files)
├── notebooks/                        # Jupyter notebooks for EDA and prototyping
├── reports/                          # Markdown/PDF reports for comparative analysis
├── src/                              # Source code for the assignment
│   ├── q1_segmentation_tagging/      # Q1: Word Segmentation & Morphology POS Tagging
│   │   ├── segmentation.py           # Viterbi segmenter with trie & language model
│   │   └── pos_tagging.py            # HMM / Viterbi morphology-aware POS tagger
│   ├── q2_dependency_parser/         # Q2: Transition-Based Dependency Parser
│   │   ├── parser.py                 # Arc-Standard parser & oracle simulator
│   │   └── feature_extraction.py     # Configuration state feature extraction
│   ├── q3_spelling_corrector/        # Q3: Efficient Spelling Corrector
│   │   ├── candidate_gen.py          # Edit distance & Symmetric Delete (SymSpell)
│   │   └── spell_check.py            # Real-word / non-word spelling corrector & CLI
│   ├── q4_integrated_editor/         # Q4: Integrated Background Editor (Streamlit)
│   │   ├── pcfg_parser.py            # CKY / PCFG constituency parser
│   │   └── streamlit_app.py          # Interactive Streamlit web interface
│   └── utils/                        # Shared utility scripts (evaluation, helpers)
├── pyproject.toml                    # Project configuration and dependencies
├── uv.lock                           # UV lockfile for reproducible environments
├── .python-version                   # Python version specification
├── .gitignore                        # Git ignore file
└── README.md                         # Project documentation
```

## Setup Instructions

### Option 1: Using `uv` (Recommended)

[`uv`](https://github.com/astral-sh/uv) is an extremely fast Python package and project manager.

1. **Install `uv`** (if not already installed):
   - **macOS / Linux**:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
   - **Windows** (PowerShell):
     ```powershell
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```
   - Alternatively via `pip` or `brew`:
     ```bash
     pip install uv
     # or on macOS:
     brew install uv
     ```

2. **Sync and Create Environment:**
   Run `uv sync` from the project root to automatically create the virtual environment and install all dependencies locked in `uv.lock`:
   ```bash
   uv sync
   ```

3. **Run Commands / Streamlit App with `uv`:**
   You can run scripts or launch the Streamlit app without manual activation:
   ```bash
   # Run the Q4 Streamlit Editor
   uv run streamlit run src/q4_integrated_editor/streamlit_app.py

   # Or run any Python script
   uv run python src/q1_segmentation_tagging/segmentation.py
   ```

---

### Option 2: Using Standard `pip` and Virtual Environment

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -e .
   ```

---

### 3. Download Datasets & Corpora
The assignment requires specific corpora. Download the Universal Dependencies datasets:
```bash
# Inside the data/raw directory:
cd data/raw
git clone https://github.com/UniversalDependencies/UD_Spanish-GSD.git
git clone https://github.com/UniversalDependencies/UD_German-GSD.git
git clone https://github.com/UniversalDependencies/UD_English-EWT.git
cd ../..
```
*Note: NLTK corpora (e.g., Brown, Treebank) are automatically downloaded or can be fetched via `nltk.download()` in the scripts.*

## Assignment Breakdown

### Question 1: Word Segmentation and POS Tagging
Implementation of a Viterbi-based dynamic programming approach with a trigram language model to segment spaceless sentences and tag them with morphology-aware POS tags (evaluating on English and Spanish/German).

### Question 2: Transition-Based Dependency Parser
Building an arc-standard dependency parser from scratch, simulating an oracle, extracting features, and training a scikit-learn classifier to predict transitions (evaluated with Labeled Attachment Score).

### Question 3: Spelling Corrector
Developing an efficient spelling corrector for non-word and real-word errors using edit distance generation and the Symmetric Delete method. Includes a "Speed Demon" benchmark and a live terminal CLI.

### Question 4: Integrated Background Editor
A Streamlit web application that simulates live typing and runs the Q1 segmentation, Q3 spelling corrector, and a PCFG Constituency parser in the background to provide real-time grammatical alerts.
