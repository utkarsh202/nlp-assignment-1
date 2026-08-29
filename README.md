# NLP Group Assignment 1

This repository contains the implementation for the NLP Group Assignment, focusing on various core Natural Language Processing tasks ranging from word segmentation to building a transition-based dependency parser and an integrated Streamlit-based editor.

## Project Structure

```text
.
├── data/                       # Dataset directory
│   ├── processed/              # Cleaned/processed datasets
│   └── raw/                    # Raw corpora (Brown, UD treebanks, CoNLL-U files)
├── notebooks/                  # Jupyter notebooks for EDA and prototyping
├── reports/                    # Markdown/PDF reports for comparative analysis
├── src/                        # Source code for the assignment
│   ├── q1_segmentation_tagging/  # Q1: Word Segmentation and POS Tagging
│   ├── q2_dependency_parser/     # Q2: Transition-Based Dependency Parser
│   ├── q3_spelling_corrector/    # Q3: Efficient Spelling Corrector
│   ├── q4_integrated_editor/     # Q4: Integrated Background Editor (Streamlit)
│   └── utils/                    # Shared utility scripts (data loading, evaluation)
├── .gitignore                  # Ignored files and folders
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

## Setup Instructions

1. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Datasets:**
   The assignment requires specific corpora. Download the Universal Dependencies and NLTK data:
   ```bash
   # Inside the data/raw directory:
   git clone https://github.com/UniversalDependencies/UD_Spanish-GSD.git
   git clone https://github.com/UniversalDependencies/UD_German-GSD.git
   git clone https://github.com/UniversalDependencies/UD_English-EWT.git
   ```
   *Note: NLTK corpora (Brown, Treebank) can be downloaded programmatically via `nltk.download()` in the scripts.*

## Assignment Breakdown

### Question 1: Word Segmentation and POS Tagging
Implementation of a Viterbi-based dynamic programming approach with a trigram language model to segment spaceless sentences and tag them with morphology-aware POS tags (evaluating on English and Spanish/German).

### Question 2: Transition-Based Dependency Parser
Building an arc-standard dependency parser from scratch, simulating an oracle, extracting features, and training a scikit-learn classifier to predict transitions (evaluated with Labeled Attachment Score).

### Question 3: Spelling Corrector
Developing an efficient spelling corrector for non-word and real-word errors using edit distance generation and the Symmetric Delete method. Includes a "Speed Demon" benchmark and a live terminal CLI.

### Question 4: Integrated Background Editor
A Streamlit web application that simulates live typing and runs the Q1 segmentation, Q3 spelling corrector, and a PCFG Constituency parser in the background to provide real-time grammatical alerts.
