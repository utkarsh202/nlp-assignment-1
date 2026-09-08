# Question 1: Word Segmentation and POS Tagging Report

## 1. Problem Statement
The objective of this assignment is to build a full NLP pipeline from scratch that performs two fundamental tasks: **Word Segmentation** and **Part-of-Speech (POS) Tagging**. 

Instead of relying on pre-built tokenizers or taggers, the pipeline uses statistical models trained on large corpora:
1. A **Trigram Language Model** combined with the **Viterbi Algorithm** to segment continuous streams of characters into words.
2. A **Second-Order Hidden Markov Model (HMM)** to predict the most likely sequence of POS tags for those words.
3. An extension to handle **Morphology-Aware Tags** (e.g., incorporating Gender and Number) to test if agreement constraints improve tagging accuracy.

The models are evaluated on two typologically different languages: **English** (Brown Corpus) and **Spanish** (Universal Dependencies GSD Treebank).

---

## 2. Implementation & Architecture

### 2.1 Word Segmentation (Trigram LM + Viterbi)
The segmenter predicts the most likely word boundaries in an unsegmented string. 
* **Language Model:** A Trigram LM is trained to calculate the probability of a word given the previous two words, $P(w_3 | w_1, w_2)$, using Add-1 (Laplace) smoothing to handle unseen n-grams.
* **Decoding:** A dynamic programming approach (Viterbi) finds the optimal path through the character sequence, constraining candidate words to those seen in the training vocabulary.

### 2.2 POS Tagging (Second-Order HMM)
The tagger assigns Universal POS tags to segmented words.
* **Emission Probabilities:** $P(word | tag)$, measuring how likely a tag generates a specific word, smoothed using Add-1.
* **Transition Probabilities:** $P(tag_3 | tag_1, tag_2)$, measuring the likelihood of a tag following the previous two tags.
* **Decoding:** A second Viterbi decoder finds the most probable sequence of tags for the given sentence.

### 2.3 Morphology-Aware Tagging
To test if agreement constraints help the HMM, we augmented the base UPOS tags.
* **Spanish:** Extracted explicit `Gender` and `Number` features from the CoNLL-U `feats` column (e.g., `NOUN-M-Sg`).
* **English:** Used surface-form suffix heuristics (e.g., `VERB-Past`, `NOUN-Pl`) since the Brown corpus lacks detailed morphological annotations.
* A separate HMM was trained on this expanded tagset.

---

## 3. Key Functions Overview

| Function | Purpose |
|----------|---------|
| `train_trigram_lm()` | Computes unigram, bigram, and trigram counts for the segmentation LM. |
| `make_log_prob()` | Returns a smoothed trigram log-probability function. |
| `viterbi_segment()` | DP algorithm to segment character strings into vocabulary words. |
| `train_hmm()` | Computes emission and second-order transition counts for POS tags. |
| `viterbi_pos()` | DP algorithm to find the most likely POS tag sequence. |
| `extract_morph_tag_from_feats()` | Augments Spanish UPOS tags with CoNLL-U gender/number features. |
| `evaluate_pipeline_errors()` | Distinguishes between genuine POS errors and segmentation-induced errors. |

---

## 4. Evaluation and Results

Both models were compared against simple baselines: a Greedy Longest-Match segmenter and a Most-Frequent-Tag (MFT) POS tagger. 

### 4.1 Comparative Analysis (English vs Spanish)
**1. Segmentation**
* The **English** Viterbi segmenter generally achieved higher F1 scores than the **Spanish** model.
* *Reason:* The English Brown Corpus training split (~45,000 sentences) is significantly larger than the Spanish GSD training split (~14,000 sentences). A robust Trigram LM requires vast amounts of data to reliably disambiguate boundary choices.

**2. POS Tagging**
* **English** HMM tagging accuracy is generally higher than **Spanish**.
* *Reason:* Spanish features richer inflectional morphology (gender, number, complex verb conjugations), resulting in a larger effective vocabulary and sparser trigram contexts for the HMM transitions.

### 4.2 Agreement-Aware Tagging: Help or Hindrance?
Expanding the tagset to include morphological features (e.g., converting 17 UPOS tags into ~80 morphology-aware tags) had mixed results:
* When data is abundant for specific constructions, the strict agreement rules (e.g., masculine-singular determiners preceding masculine-singular nouns) **helped** resolve ambiguities.
* However, overall accuracy often **dropped slightly** on the Spanish dev set. Expanding the tagset splits the probability mass, severely increasing data sparsity. The second-order transition model $P(t_3 | t_1, t_2)$ struggles to learn reliable probabilities for rare morphology combinations without a massive corpus.

### 4.3 Error Source Analysis
Running the full pipeline (Segment $\rightarrow$ Tag) reveals two distinct error sources:
1. **Segmentation-induced errors:** If a word boundary is wrong, the resulting sub-words will almost certainly receive incorrect POS tags.
2. **Genuine POS errors:** The word was correctly segmented, but assigned the wrong tag.

*Result:* A significant proportion of final POS errors (often the majority) are actually **segmentation-induced**. This proves that in pipelined NLP systems, upstream errors cascade heavily, and investing in better segmentation yields compounding returns.

---

## 5. Conclusion
The implementation fully addresses the requirements of Question 1. The Trigram Viterbi Segmenter drastically reduces the boundary errors produced by the greedy baseline, while the Second-Order HMM captures syntactic structures that simple frequency baselines ignore. The morphological experiments effectively demonstrated the trade-off in NLP between expressivity (richer tags) and learnability (data sparsity).
