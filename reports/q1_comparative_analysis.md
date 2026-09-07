# Question 1: Comparative Analysis Report
## Word Segmentation and Morphology-Aware POS Tagging

---

## 1. Executive Summary & Experimental Setup

This report presents the implementation and evaluation of a joint pipeline for **Word Segmentation** and **Part-of-Speech (POS) Tagging** for English and Spanish.

### English
- Dataset: **NLTK Brown Corpus**
- Approximately 12,000 sentences were used.
- The original training portion was further divided into training and development sets.
- The remaining portion was used for testing.

### Spanish
- Dataset: **Universal Dependencies Spanish-GSD Treebank**
- Official train, development, and test splits were used.

The system consists of:

1. **Word Segmentation:** Interpolated Trigram Language Model with Viterbi dynamic programming.
2. **POS Tagging:** Trigram HMM using emission and transition probabilities.
3. **Morphology-Aware Tagging:** Spanish gender and number information is incorporated into extended POS tags.
4. **Baselines:** Greedy longest-match segmentation and most-frequent-tag POS tagging.
5. **Evaluation:** Accuracy, OOV accuracy, segmentation F1, exact sentence match, confusion matrices, and error-source analysis.

---

# 2. Word Segmentation

The input to the segmentation system is a sentence with all spaces removed.

The system searches for the most probable sequence of words using an interpolated Trigram Language Model:

\[
P(w_i|w_{i-2},w_{i-1})
\]

The model combines trigram, bigram, unigram, and uniform probabilities.

Viterbi dynamic programming is then used to find the highest-scoring segmentation.

## Segmentation Results

| Language | Model | Boundary F1 (%) | Exact Sentence Match (%) |
| :--- | :--- | :---: | :---: |
| **English** | Greedy Longest-Match Baseline | 81.61% | 11.67% |
| **English** | **Viterbi Trigram LM** | **88.94%** | **20.83%** |
| **Spanish** | Greedy Longest-Match Baseline | 73.02% | 2.50% |
| **Spanish** | **Viterbi Trigram LM** | **90.57%** | **14.17%** |

### Improvement over the baseline

**English:**

\[
88.94 - 81.61 = \mathbf{+7.33}
\]

percentage points in Boundary F1.

**Spanish:**

\[
90.57 - 73.02 = \mathbf{+17.55}
\]

percentage points in Boundary F1.

The Viterbi Trigram LM performs substantially better than greedy longest-match segmentation because it considers the probability of the complete sequence of words rather than making an independent longest-match decision at every character position.

The improvement is especially large for Spanish, where greedy segmentation achieves only 73.02% F1 while the Viterbi model reaches 90.57%.

---

# 3. POS Tagging

The POS tagger is implemented as a **Trigram Hidden Markov Model**.

The transition model estimates:

\[
P(t_i|t_{i-2},t_{i-1})
\]

and the emission model estimates the probability of a word given its POS tag.

Viterbi decoding is used to find the most probable sequence of POS tags.

## POS Results

| Language | Model | Overall Accuracy (%) | OOV Accuracy (%) |
| :--- | :--- | :---: | :---: |
| **English** | Most Frequent Tag Baseline | 86.05% | 27.27% |
| **English** | **Trigram HMM (Viterbi)** | **93.39%** | **83.06%** |
| **Spanish** | Most Frequent Tag Baseline | 88.08% | 21.00% |
| **Spanish** | **Trigram HMM (Viterbi)** | **94.31%** | **83.99%** |

### Improvement over the baseline

For English:

\[
93.39 - 86.05 = \mathbf{+7.34}
\]

percentage points.

For Spanish:

\[
94.31 - 88.08 = \mathbf{+6.23}
\]

percentage points.

The largest improvement is seen for unseen/OOV words.

### OOV performance

| Language | Baseline OOV | HMM OOV | Improvement |
| :--- | :---: | :---: | :---: |
| English | 27.27% | **83.06%** | **+55.79 points** |
| Spanish | 21.00% | **83.99%** | **+62.99 points** |

This shows that the HMM and morphological feature model provide much better information for handling words that were not directly observed during training.

---

# 4. Spanish Morphology-Aware POS Tagging

Spanish has richer gender and number morphology than English.

For this experiment, the POS tags were extended with gender and number information where available.

Examples include:

```text
DET-Fem-Sing
NOUN-Masc-Plur
ADJ-Fem-Sing
````

The morphology-aware model uses an expanded tagset of **85 tags**, compared with **17 standard UPOS tags**.

## Results

| Spanish Model             | Overall Accuracy (%) | Known Accuracy (%) | OOV Accuracy (%) |
| :------------------------ | :------------------: | :----------------: | :--------------: |
| Plain UPOS Trigram HMM    |      **94.31%**      |          —         |    **83.99%**    |
| Morphology/Agrreement HMM |      **92.30%**      |     **93.60%**     |    **78.14%**    |

The morphology-aware model has lower raw accuracy than the plain 17-tag model.

This is expected because extending the tagset from 17 to 85 categories makes the prediction task more difficult and introduces additional data sparsity.

However, the morphology-aware model explicitly represents grammatical gender and number, allowing agreement relationships such as:

```text
DET-Fem-Sing → NOUN-Fem-Sing → ADJ-Fem-Sing
```

to be represented by the transition model.

Therefore, the morphology experiment demonstrates the trade-off between **richer linguistic representation** and **prediction difficulty**.

---

# 5. Baseline Comparison

## Segmentation

The Viterbi Trigram LM clearly outperforms greedy longest-match segmentation.

### English

```text
Greedy F1  = 81.61%
Viterbi F1 = 88.94%
Improvement = +7.33 percentage points
```

### Spanish

```text
Greedy F1  = 73.02%
Viterbi F1 = 90.57%
Improvement = +17.55 percentage points
```

The Viterbi approach is able to use contextual word probabilities to select better boundaries.

## POS Tagging

### English

```text
MFT Baseline = 86.05%
Trigram HMM  = 93.39%
Improvement  = +7.34 percentage points
```

### Spanish

```text
MFT Baseline = 88.08%
Trigram HMM  = 94.31%
Improvement  = +6.23 percentage points
```

The HMM uses both lexical emission information and tag-transition context, whereas the baseline assigns the most frequent tag and therefore cannot effectively model sentence context.

---

# 6. Development-Set Beam Tuning

The POS decoder uses beam-pruned Viterbi decoding.

Beam widths of 20, 40, and 60 were evaluated on the development set.

## English

| Beam Width | Development Accuracy |
| :--------: | :------------------: |
|     20     |        91.49%        |
|     40     |        91.49%        |
|     60     |        91.49%        |

Selected beam width: **20**

## Spanish Plain POS

| Beam Width | Development Accuracy |
| :--------: | :------------------: |
|     20     |        93.95%        |
|     40     |        93.95%        |
|     60     |        93.95%        |

Selected beam width: **20**

## Spanish Morphology-Aware POS

| Beam Width | Development Accuracy |
| :--------: | :------------------: |
|     20     |        92.02%        |
|     40     |        92.02%        |
|     60     |        92.02%        |

Selected beam width: **20**

The three tested beam widths produced the same development accuracy on these evaluation subsets, so the smallest beam width, 20, was selected.

---

# 7. Error-Source Analysis

The complete pipeline consists of:

```text
Spaceless Input
      ↓
Word Segmentation
      ↓
POS Tagging
      ↓
Final Tagged Output
```

Errors were separated into:

1. **Segmentation-Induced Errors:** Incorrect word boundaries caused the downstream POS tagger to receive an incorrect token.
2. **Genuine POS Errors:** The word was correctly segmented but the POS tag was incorrect.

## Results

| Metric                      |    English   |    Spanish   |
| :-------------------------- | :----------: | :----------: |
| Total Test Tokens           |     4,040    |     3,951    |
| Pipeline Accuracy           |  **84.18%**  |  **85.35%**  |
| Segmentation-Induced Errors | 395 (61.82%) | 393 (67.88%) |
| Genuine POS Errors          | 244 (38.18%) | 186 (32.12%) |

The majority of errors in both languages are caused by segmentation.

For English:

$$
61.82\%
$$

of errors were segmentation-induced.

For Spanish:

$$
67.88\%
$$

of errors were segmentation-induced.

This shows that correct word boundaries are a major prerequisite for accurate POS tagging.

---

# 8. Confusion Matrix

## English

Top confused Brown POS tags:

| Actual \ Predicted | NN | NN-TL | IN | RB | CS | VB |
| :----------------- | -: | ----: | -: | -: | -: | -: |
| NN                 |  0 |    27 |  0 |  0 |  0 |  4 |
| NN-TL              | 18 |     0 |  0 |  0 |  0 |  0 |
| IN                 |  0 |     0 |  0 |  2 |  3 |  0 |
| RB                 |  0 |     0 |  0 |  0 |  4 |  0 |
| CS                 |  0 |     0 |  6 |  1 |  0 |  0 |
| VB                 | 10 |     1 |  1 |  0 |  0 |  0 |

## Spanish

Top confused UPOS tags:

| Actual \ Predicted | NOUN | PROPN | ADJ | PRON | VERB | SCONJ |
| :----------------- | ---: | ----: | --: | ---: | ---: | ----: |
| NOUN               |    0 |    24 |  12 |    0 |    6 |     0 |
| PROPN              |   28 |     0 |   7 |    0 |    1 |     0 |
| ADJ                |    6 |     2 |   0 |    0 |    2 |     0 |
| PRON               |    1 |     0 |   0 |    0 |    0 |     4 |
| VERB               |    3 |     0 |   2 |    0 |    0 |     0 |
| SCONJ              |    0 |     0 |   0 |    2 |    0 |     0 |

The confusion matrices show that errors occur between several linguistically or distributionally similar categories, including nouns/proper nouns, nouns/adjectives, and related function-word categories.

---

# 9. Required Sample Test Strings

## English

### Input

```text
thequickbrownfoxjumpsoverthelazydog
```

### Segmentation

```text
['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
```

### POS Tags

```text
[
    ('the', 'AT'),
    ('quick', 'JJ'),
    ('brown', 'JJ'),
    ('fox', 'NN'),
    ('jumps', 'NNS'),
    ('over', 'IN'),
    ('the', 'AT'),
    ('lazy', 'JJ'),
    ('dog', 'NN')
]
```

---

## Spanish Sample 1

### Input

```text
mispadrespuedenviajar
```

### Segmentation

```text
['mis', 'padres', 'pueden', 'viajar']
```

### Plain UPOS

```text
[
    ('mis', 'DET'),
    ('padres', 'NOUN'),
    ('pueden', 'AUX'),
    ('viajar', 'VERB')
]
```

### Agreement Tags

```text
[
    ('mis', 'DET-Plur'),
    ('padres', 'NOUN-Masc-Plur'),
    ('pueden', 'AUX-Plur'),
    ('viajar', 'VERB')
]
```

---

## Spanish Sample 2

### Input

```text
elcielodespejadoesazul
```

### Segmentation

```text
['el', 'cielo', 'despejado', 'es', 'azul']
```

### Plain UPOS Model Output

```text
[
    ('el', 'DET'),
    ('cielo', 'NOUN'),
    ('despejado', 'VERB'),
    ('es', 'AUX'),
    ('azul', 'ADJ')
]
```

### Agreement Demonstration

The expected morphology-aware interpretation demonstrates the gender/number structure:

```text
[
    ('el', 'DET-Masc-Sing'),
    ('cielo', 'NOUN-Masc-Sing'),
    ('despejado', 'ADJ-Masc-Sing'),
    ('es', 'AUX-Sing'),
    ('azul', 'ADJ-Sing')
]
```

---

## Spanish Sample 3

### Input

```text
lacasarojaesgrande
```

### Segmentation

```text
['la', 'casa', 'roja', 'es', 'grande']
```

### Plain UPOS

```text
[
    ('la', 'DET'),
    ('casa', 'NOUN'),
    ('roja', 'ADJ'),
    ('es', 'AUX'),
    ('grande', 'ADJ')
]
```

### Agreement Tags

```text
[
    ('la', 'DET-Fem-Sing'),
    ('casa', 'NOUN-Fem-Sing'),
    ('roja', 'ADJ-Fem-Sing'),
    ('es', 'AUX-Sing'),
    ('grande', 'ADJ-Sing')
]
```

---

# 10. Overall Comparison

| Component                    |   English  |   Spanish  |
| :--------------------------- | :--------: | :--------: |
| Greedy Segmentation F1       |   81.61%   |   73.02%   |
| Viterbi Segmentation F1      | **88.94%** | **90.57%** |
| MFT POS Accuracy             |   86.05%   |   88.08%   |
| Trigram HMM POS Accuracy     | **93.39%** | **94.31%** |
| Pipeline Accuracy            |   84.18%   |   85.35%   |
| Segmentation-Induced Error % |   61.82%   |   67.88%   |

Spanish benefits particularly strongly from probabilistic segmentation, with a **17.55 percentage-point improvement** over the greedy baseline.

Both languages show substantial improvements from the trigram HMM over the most-frequent-tag baseline.

---

# 11. Conclusion

The implementation demonstrates that probabilistic sequence models provide substantial improvements over simple greedy and frequency-based baselines.

For word segmentation, the Viterbi Trigram Language Model improved Boundary F1 from **81.61% to 88.94%** for English and from **73.02% to 90.57%** for Spanish.

For POS tagging, the Trigram HMM improved accuracy from **86.05% to 93.39%** for English and from **88.08% to 94.31%** for Spanish.

The OOV results also demonstrate the benefit of using learned emission and morphological information instead of relying only on the most frequent tag.

Error analysis showed that segmentation mistakes account for the majority of downstream pipeline errors in both languages. This indicates that improving word boundaries is important for reliable POS tagging.

The Spanish morphology-aware experiment demonstrates the trade-off between richer linguistic representation and prediction difficulty. Extending the tagset with gender and number provides explicit agreement information, but the larger 85-tag space introduces additional sparsity and reduces raw accuracy compared with the simpler 17-tag UPOS model.

Overall, the implementation covers the required components of Question 1: probabilistic word segmentation, trigram POS tagging, morphology-aware Spanish tagging, baseline comparison, development-set tuning, quantitative evaluation, confusion-matrix analysis, error-source separation, and required sample demonstrations.
