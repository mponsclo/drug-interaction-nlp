# Biomedical NLP: Drug NER & Drug-Drug Interaction Detection

> Identifying drug names and detecting pharmacological interactions in biomedical text using three incremental approaches: rule-based heuristics, classical machine learning, and deep learning.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![CRFsuite](https://img.shields.io/badge/CRFsuite-sequence%20labeling-green.svg)](https://python-crfsuite.readthedocs.io/)

## Pipeline Overview

| Task | Rule-Based | Classical ML | Deep Learning |
|------|-----------|-------------|---------------|
| **NER** (drug name recognition) | Suffix/prefix heuristics + dictionary lookup | CRF with lexical and contextual features | BiLSTM with word embeddings |
| **DDI** (drug-drug interaction) | Dependency-tree patterns (stub) | MaxEnt with syntactic path features | LSTM + Conv1D hybrid |

## Project Structure

```
.
├── ner/                        # Named Entity Recognition
│   ├── baseline.py             #   Rule-based: suffix patterns + DrugBank lookup
│   ├── feature_extractor.py    #   CRF: 18+ features per token (POS, lemma, affixes)
│   ├── crf_learner.py          #   CRF: model training with L1/L2 regularization
│   ├── crf_classifier.py       #   CRF: BIO tag prediction and entity assembly
│   ├── utils.py                #   Feature file parsing utilities
│   └── notebooks/              #   EDA, baseline analysis, CRF experimentation
├── ddi/                        # Drug-Drug Interaction Detection
│   ├── baseline.py             #   Rule-based scaffold (stub)
│   ├── feature_extractor.py    #   MaxEnt: dependency path + syntactic features
│   ├── learner.py              #   MaxEnt: MEGAM-backed classifier training
│   ├── utils.py                #   Feature file parsing utilities
│   ├── runner.sh               #   End-to-end pipeline script
│   └── notebooks/              #   EDA, baseline analysis, MaxEnt experimentation
├── neural/                     # Deep Learning (Keras/TensorFlow)
│   └── notebooks/
│       ├── 01_ner_nn.ipynb     #   BiLSTM for drug name recognition
│       └── 02_ddi_nn.ipynb     #   LSTM+Conv1D for interaction classification
├── shared/
│   └── evaluator.py            # Unified evaluation (precision, recall, F1)
├── resources/                  # Drug name databases (DrugBank, HSDB)
└── reports/                    # Technical reports (PDF)
```

## Quick Start

### Installation

```bash
git clone https://github.com/mponsclo/Advanced_Human_Language_Technologies.git
cd Advanced_Human_Language_Technologies
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

### NER: Rule-Based Baseline

```bash
cd ner
python baseline.py data_dir output.txt           # suffix heuristics only
python baseline.py -l data_dir output.txt         # with dictionary lookup
```

### NER: CRF Pipeline

```bash
cd ner
python feature_extractor.py train_dir train_feats.txt
python feature_extractor.py test_dir test_feats.txt
python crf_learner.py model.crf train_feats.txt
python crf_classifier.py model.crf test_feats.txt
```

### DDI: Full Pipeline

```bash
cd ddi
DATA_DIR=/path/to/semeval/data bash runner.sh
```

Requires a running [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) server:

```bash
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

### Neural Networks

Open the notebooks in `neural/notebooks/` with Jupyter or Google Colab. Each notebook is self-contained with data loading, model definition, training, and evaluation.

## Methodology

### 1. Rule-Based Baseline (NER)

Classifies tokens using pharmacological suffix patterns and dictionary lookup:

- **Suffix matching**: 3/4/5-letter suffixes (*-azole*, *-mycin*, *-arin*, *-lol*, ...) map to `drug`; group keywords (*inhibitor*, *steroid*, *NSAID*, ...) map to `group`; all-caps >=4 chars map to `brand`
- **Dictionary lookup** (optional `-l` flag): checks tokens against DrugBank (~3.9 MB, categorized as drug/brand/group) and HSDB (~4,700 entries)
- **Bigram assembly**: consecutive tokens separated by exactly 2 characters are joined and classified as a single entity
- Stopwords and non-alphabetic tokens are filtered before classification

### 2. CRF Sequence Labeling (NER)

BIO-tagged Conditional Random Fields with 25+ handcrafted features per token:

| Feature Group | Features |
|--------------|----------|
| Surface | word form, lowercase, length |
| Morphology | 3/4/5-char suffixes and prefixes |
| Lexical | POS tag, lemma, digit count, contains dash |
| Boolean | is capitalized, is uppercase, is digit, is stopword, is punctuation |
| Context | previous/next token form, suffixes, and boolean properties |
| Dictionary | DrugBank/HSDB lookup result (optional) |

- **CRF hyperparameters**: L1=0.2, L2=0.001, max iterations=1000
- Sequence-aware: captures entity boundaries via B-I-O transitions

### 3. MaxEnt Classifier (DDI)

Maximum Entropy model over syntactic features extracted from Stanford CoreNLP dependency parses:

- **Dependency path**: shortest path between entity pair in the undirected dependency graph (via networkx)
- **Path traversal**: encodes the sequence of dependency relations and POS tags from each entity up to their common ancestor
- **Head features**: lemma and POS tag of each entity's syntactic head; whether entities share a head; whether shared head is a verb
- **Clue verbs**: presence of *administer*, *enhance*, *interact*, *coadminister*, *increase*, *decrease* on the path
- **Negation signals**: count of negative words (*not*, *without*, *prevent*, *unlikely*, ...) in path and full sentence
- **Context**: lemmas and POS tags of words outside the dependency path; types of other entities in the sentence
- Classifies pairs as: `effect`, `mechanism`, `advice`, `int`, or `null`

### 4. Deep Learning

#### NER: Multi-Input BiLSTM (2M parameters)

Three parallel branches (word, suffix, POS tag), each with a 64-dim embedding + BiLSTM (128 units), concatenated into a final BiLSTM (64) + BiRNN (64) + Dense (10, softmax). Uses pre-trained GloVe 6B embeddings. Trained with Adam (lr=0.005, amsgrad).

#### DDI: BiLSTM + Conv1D Hybrid (697K parameters)

Three branches (word, suffix, prefix) with BiLSTM (32 units each), concatenated into BiLSTM (64) + Conv1D (128 filters, kernel=5) + GlobalMaxPool + Dense (5, softmax). GloVe 16-dim word embeddings. Trained with Adam (lr=0.005) for 5 epochs.

## Results

| Task | Method | Metric | Score |
|------|--------|--------|-------|
| DDI | BiLSTM + Conv1D | Validation accuracy | 87.4% |
| DDI | BiLSTM + Conv1D | Test macro-F1 | 42.2% |
| DDI | BiLSTM + Conv1D | Best class F1 (effect) | 57.4% |

*NER metrics are generated at runtime via the evaluator; see notebooks for detailed per-class breakdowns.*

## Data

This project uses the [SemEval-2013 Task 9: DDIExtraction](https://www.cs.york.ac.uk/semeval-2013/task9/) dataset (~11,600 drug entities, ~23,100 DDI training pairs across 4 interaction types). The data is not included in this repository -- obtain it separately and point scripts to it via the `data_dir` argument or `DATA_DIR` environment variable.

## Academic Context

**Course**: Advanced Human Language Technologies
**Institution**: Universitat Politecnica de Catalunya (UPC)
