# Question 1: Comparative Analysis Report
**Word Segmentation and Morphology-Aware POS Tagging**

---

## 1. Executive Summary & Experimental Setup
This report analyzes the performance of a joint pipeline for **Word Segmentation** and **Part-of-Speech (POS) Tagging** across two distinct linguistic topologies:
- **English**: Analytic language with minimal inflectional morphology, evaluated on the **NLTK Brown Corpus** (80/20 train/test split, ~12,000 sentences).
- **Spanish**: Synthetic/fusional Romance language with rich gender/number inflection and agreement patterns, evaluated on the **Universal Dependencies Spanish-GSD Treebank** (`es_gsd-ud-train.conllu` and `es_gsd-ud-test.conllu`).

The pipeline first segments spaceless continuous letter sequences using an interpolated **Trigram Language Model with Viterbi dynamic programming search**, followed by **Hidden Markov Model (HMM) POS Tagging** incorporating subword morphological feature modeling and gender/number agreement tagging.

---

## 2. Experimental Results Summary

### A. Word Segmentation Performance
| Language | Model | Boundary Prec (%) | Boundary Rec (%) | Boundary F1 (%) | Exact Sentence Match (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **English** | Greedy Longest-Match Baseline | 76.81% | 87.89% | 81.97% | 11.67% |
| **English** | **Viterbi Trigram LM** | **84.54%** | **95.94%** | **89.88%** | **23.33%** |
| **Spanish** | Greedy Longest-Match Baseline | 64.92% | 83.41% | 73.02% | 2.50% |
| **Spanish** | **Viterbi Trigram LM** | **85.34%** | **96.50%** | **90.57%** | **14.17%** |

### B. POS Tagging Performance
| Language | Tagset Level | Model | Overall Acc (%) | Known Words Acc (%) | Unseen/OOV Acc (%) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **English** | Brown POS (281 tags) | Most Frequent Tag Baseline | 86.87% | 91.24% | 26.98% |
| **English** | Brown POS (281 tags) | **Morphology HMM (Viterbi)** | **93.96%** | **94.73%** | **83.49%** |
| **Spanish** | UPOS (17 tags) | Most Frequent Tag Baseline | 88.08% | 91.13% | 21.00% |
| **Spanish** | UPOS (17 tags) | **Morphology HMM (Viterbi)** | **94.43%** | **94.92%** | **83.65%** |
| **Spanish** | Agreement (`UPOS-Gen-Num`, 85 tags) | Most Frequent Tag Baseline | 82.15% | 86.30% | 15.42% |
| **Spanish** | Agreement (`UPOS-Gen-Num`, 85 tags) | **Agreement HMM (Viterbi)** | **92.50%** | **93.79%** | **78.49%** |

### C. Error-Source Separation (Part 5)
When passing spaceless input through the combined pipeline, errors in final token labeling originate from two separate sources:
1. **Segmentation-Induced Errors**: Word boundary was incorrectly placed, causing the downstream tagger to operate on an invalid token.
2. **Genuine POS Tagging Errors**: Word boundary was segmented perfectly, but the tagger assigned the wrong grammatical category.

| Metric | English Pipeline | Spanish Pipeline |
| :--- | :---: | :---: |
| **Total Test Tokens Evaluated** | 4,040 | 3,951 |
| **End-to-End Pipeline Accuracy** | **86.24%** | **85.60%** |
| **Total Errors** | 556 | 569 |
| **Segmentation-Induced Errors** | **351 (63.13%)** | **393 (69.07%)** |
| **Genuine POS Tagging Errors** | **205 (36.87%)** | **176 (30.93%)** |

---

## 3. Detailed Answers to Core Assignment Questions

### Question A: Where did English and Spanish differ most in accuracy?
1. **Segmentation Sensitivity**:
   - The greedy baseline suffered far more severely in Spanish (**73.02% F1, 2.50% exact match**) than in English (**81.97% F1, 11.67% exact match**). Spanish contains numerous short function words (e.g., prepositions `a`, `de`, `en`, articles `el`, `la`, pronouns `se`, `me`) that frequently prefix longer nouns and verbs (e.g., `despejado` starting with `de`). Greedy matching greedily consumes subwords, inducing catastrophic error cascades.
   - The Viterbi Trigram LM leveled the playing field, reaching **>90% F1** in both languages by evaluating global sentence probabilities.
2. **Tag Granularity**:
   - In standard UPOS (17 tags), Spanish achieved higher overall tagging accuracy (**94.43%**) than English on Brown tags (**93.96%**), primarily because Brown tags are fine-grained (281 distinct syntactic tags).

---

### Question B: Did agreement-aware tagging actually help, or add noise?
- In Part 3, we extended the Spanish tagset from 17 universal tags to 85 morphology-aware tags capturing gender and number (e.g., `DET-Fem-Sing`, `NOUN-Fem-Sing`, `ADJ-Fem-Sing`).
- **Linguistic Insight**:
  - The model successfully captured grammatical agreement: feminine singular determiners (`la`) transition with high probability to feminine singular nouns (`casa`), which transition to feminine singular adjectives (`roja`).
  - This is evidenced by our sample test runs:
    - `elcielodespejadoesazul` $\to$ `[('el', 'DET-Masc-Sing'), ('cielo', 'NOUN-Masc-Sing'), ('despejado', 'ADJ-Masc-Sing'), ('es', 'AUX-Sing'), ('azul', 'ADJ-Sing')]`
    - `lacasarojaesgrande` $\to$ `[('la', 'DET-Fem-Sing'), ('casa', 'NOUN-Fem-Sing'), ('roja', 'ADJ-Fem-Sing'), ('es', 'AUX-Sing'), ('grande', 'ADJ-Sing')]`
- **Quantitative Trade-off**:
  - Expanding the tag space from 17 to 85 tags caused a slight drop in raw token accuracy (**94.43% $\to$ 92.50%**), which is expected due to data sparsity in joint category prediction.
  - However, the **Agreement HMM significantly outperformed the Most Frequent Tag baseline** on the same tagset (**92.50% vs 82.15%**, a +10.35% absolute gain), proving that the transition matrix successfully exploited syntactic agreement constraints rather than merely adding random noise.

---

### Question C: How much of the tagging error came from segmentation mistakes vs. genuine tagging mistakes?
- In both languages, **the majority of pipeline tagging errors (~63% to ~69%) were caused by upstream segmentation mistakes**, rather than failure of the POS tagger:
  - **English**: 63.13% segmentation-induced vs. 36.87% genuine POS errors.
  - **Spanish**: 69.07% segmentation-induced vs. 30.93% genuine POS errors.
- **Key Takeaway**: POS tagging is heavily bottlenecked by word boundary integrity. When a word boundary is misaligned by even a single character, the resulting non-word token triggers an OOV emission failure, almost guaranteeing a POS misclassification.

---

### Question D: How much better were your models than the simple baselines?
1. **Segmentation (Trigram Viterbi vs. Greedy Longest-Match)**:
   - English: +7.91% Boundary F1; Exact sentence recovery doubled (**11.67% $\to$ 23.33%**).
   - Spanish: **+17.55% Boundary F1**; Exact sentence recovery improved **more than 5.6-fold** (**2.50% $\to$ 14.17%**).
2. **POS Tagging (Morphology HMM vs. Most Frequent Tag)**:
   - Known Word Accuracy: Improved by +3.5% to +7.5%.
   - **Out-of-Vocabulary (OOV) Word Accuracy**:
     - English: **26.98% $\to$ 83.49% (+56.51% absolute gain)**.
     - Spanish (UPOS): **21.00% $\to$ 83.65% (+62.65% absolute gain)**.
     - Spanish (Agreement): **15.42% $\to$ 78.49% (+63.07% absolute gain)**.
   - The naive baseline blindly predicts the majority class (`NOUN`) for unseen words, whereas our morphology model leverages prefix, suffix, and shape distributions (e.g., `-ly` $\to$ `ADV`, `-ción` $\to$ `NOUN`, `-ar`/`-ando` $\to$ `VERB`, `-mente` $\to$ `ADV`).

---

## 4. Qualitative Outputs on Required Sample Test Strings

### 1. English
- **Input**: `thequickbrownfoxjumpsoverthelazydog`
- **Segmented Words**: `['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']`
- **Predicted POS Tags**:
  `[('the', 'AT'), ('quick', 'JJ'), ('brown', 'JJ'), ('fox', 'NN'), ('jumps', 'VBZ'), ('over', 'IN'), ('the', 'AT'), ('lazy', 'JJ'), ('dog', 'NN')]`

### 2. Spanish Sample 1
- **Input**: `mispadrespuedenviajar`
- **Segmented Words**: `['mis', 'padres', 'pueden', 'viajar']`
- **Plain UPOS**: `[('mis', 'DET'), ('padres', 'NOUN'), ('pueden', 'AUX'), ('viajar', 'VERB')]`
- **Agreement Tags**: `[('mis', 'DET-Plur'), ('padres', 'NOUN-Masc-Plur'), ('pueden', 'AUX-Plur'), ('viajar', 'VERB')]`

### 3. Spanish Sample 2
- **Input**: `elcielodespejadoesazul`
- **Segmented Words**: `['el', 'cielo', 'despejado', 'es', 'azul']`
- **Plain UPOS**: `[('el', 'DET'), ('cielo', 'NOUN'), ('despejado', 'ADJ'), ('es', 'AUX'), ('azul', 'ADJ')]`
- **Agreement Tags**: `[('el', 'DET-Masc-Sing'), ('cielo', 'NOUN-Masc-Sing'), ('despejado', 'ADJ-Masc-Sing'), ('es', 'AUX-Sing'), ('azul', 'ADJ-Sing')]`

### 4. Spanish Sample 3
- **Input**: `lacasarojaesgrande`
- **Segmented Words**: `['la', 'casa', 'roja', 'es', 'grande']`
- **Plain UPOS**: `[('la', 'DET'), ('casa', 'NOUN'), ('roja', 'ADJ'), ('es', 'AUX'), ('grande', 'ADJ')]`
- **Agreement Tags**: `[('la', 'DET-Fem-Sing'), ('casa', 'NOUN-Fem-Sing'), ('roja', 'ADJ-Fem-Sing'), ('es', 'AUX-Sing'), ('grande', 'ADJ-Sing')]`
