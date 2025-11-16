# Natural Language Processing - UT Austin MS AI Program

This repository contains coursework for the Natural Language Processing course in the UT Austin Master of Science in Artificial Intelligence program. The assignments progress from classical ML approaches to modern deep learning techniques for NLP tasks.

## Table of Contents
- [Environment Setup](#environment-setup)
- [Assignment 0: Introduction to NLP](#assignment-0-introduction-to-nlp)
- [Assignment 1: Sentiment Classification](#assignment-1-sentiment-classification)
- [Assignment 2: Feedforward Neural Networks & Word Embeddings](#assignment-2-feedforward-neural-networks--word-embeddings)
- [Assignment 3: Transformer Language Modeling](#assignment-3-transformer-language-modeling)
- [Assignment 4: Fact-checking ChatGPT Outputs](#assignment-4-fact-checking-chatgpt-outputs)
- [Final Project: NLI and QA with ELECTRA](#final-project-nli-and-qa-with-electra)
- [Course Information](#course-information)

---

## Environment Setup

### Requirements
- Python 3.5+ (Python 3.10 recommended for Assignment 4)
- PyTorch (for Assignments 2-4)
- NumPy
- Additional packages: nltk, spacy (see individual assignment requirements)

### Installation
```bash
# Using Anaconda (recommended)
conda create -n nlp_env python=3.10
conda activate nlp_env

# Install PyTorch
# Visit https://pytorch.org/get-started/locally/ for system-specific instructions

# Install other dependencies (from assignment directories)
pip install -r requirements.txt
```

---

## Assignment 0: Introduction to NLP

**Location:** `Assignment 0/`

Basic introduction to text processing and tokenization.

### Files
- `tokenizer.py` - Basic tokenization implementation
- `count.py` - Word counting utilities
- `token_counter.py` - Token statistics

---

## Assignment 1: Sentiment Classification

**Location:** `Assignment 1/a1-distrib/`

**Objective:** Implement classical ML classifiers for binary sentiment classification on movie reviews.

### Dataset
- **Source:** Rotten Tomatoes movie review dataset (Socher et al., 2013)
- **Task:** Binary sentiment classification (positive/negative)
- **Format:** Tab-separated label and tokenized sentence

### Implementation Tasks

#### Part 1: Perceptron (40 points)
- Implement perceptron classifier with bag-of-words unigram features
- **Target:** ≥74% dev set accuracy
- **Runtime:** <20 seconds

**Key Components:**
- `UnigramFeatureExtractor` - Extract unigram features
- `PerceptronClassifier` - Perceptron inference
- `train_perceptron` - Training loop

#### Part 2: Logistic Regression (30 points)
- Implement logistic regression with unigram features
- **Target:** ≥77% dev set accuracy
- **Runtime:** <20 seconds

**Key Components:**
- `LogisticRegressionClassifier`
- `train_logistic_regression`

#### Part 3: Feature Engineering (30 points)
- **Q3:** Implement `BigramFeatureExtractor` (15 points)
- **Q4:** Implement `BetterFeatureExtractor` with advanced features (15 points)
  - Options: n-grams, tf-idf weighting, stopword removal, etc.
  - **Runtime:** <60 seconds

### Running the Code
```bash
cd "Assignment 1/a1-distrib"

# Perceptron with unigrams
python sentiment_classifier.py --model PERCEPTRON --feats UNIGRAM

# Logistic regression with unigrams
python sentiment_classifier.py --model LR --feats UNIGRAM

# Logistic regression with bigrams
python sentiment_classifier.py --model LR --feats BIGRAM

# Best feature extractor
python sentiment_classifier.py --model LR --feats BETTER
```

### Files
- `sentiment_classifier.py` - Main driver (do not modify for submission)
- `models.py` - Implement classifiers and feature extractors here
- `sentiment_data.py` - Data loading utilities
- `utils.py` - Indexer class for feature mapping

---

## Assignment 2: Feedforward Neural Networks & Word Embeddings

**Location:** `Assignment 2/a2-distrib/`

**Objective:** Build deep averaging networks with pre-trained word embeddings, explore optimization and generalization.

### Dataset
- Same sentiment classification dataset from Assignment 1 (lowercased)
- Pre-trained GloVe embeddings (50d and 300d)

### Implementation Tasks

#### Part 1: Optimization (25 points)
- Implement gradient descent for quadratic function
- Find optimal step size

**Files:** `optimization.py`

```bash
python optimization.py --lr 1
```

#### Part 2: Deep Averaging Network (75 points)

**Q2a (50 points):** Implement DAN
- Average word embeddings as input
- Feedforward network for classification
- **Target:** ≥77% dev accuracy
- **Runtime:** <10 minutes

**Q2b:** Implement batching for efficiency

**Q3 (25 points):** Handle typos
- Dataset: `dev-typo.txt` with random misspellings
- **Target:** ≥74% accuracy on typo data

**Options:**
1. Spelling correction with edit distance
2. Prefix embeddings (first 3 characters)
3. Custom solution

### Running the Code
```bash
cd "Assignment 2/a2-distrib"

# Optimization
python optimization.py

# DAN training
python neural_sentiment_classifier.py

# Typo-aware model
python neural_sentiment_classifier.py --use_typo_setting

# Use 50d embeddings for faster debugging
python neural_sentiment_classifier.py --word_vecs_path data/glove.6B.50d-relativized.txt
```

### Key Components
- `models.py` - Implement DAN architecture
- `train_deep_averaging_network` - Training loop
- `neural_sentiment_classifier.py` - Main driver
- Example code: `ffnn_example.py` (Module 2)

---

## Assignment 3: Transformer Language Modeling

**Location:** `Assignment 3/a3-distrib/`

**Objective:** Implement Transformer encoder from scratch and apply it to language modeling.

### Dataset
- **text8:** First 100M characters from Wikipedia
- Only 27 character types (a-z + space)
- Sequences of length 20

### Implementation Tasks

#### Part 1: Transformer Encoder (50 points)

**Task:** Count character occurrences
- Given a sequence, predict how many times each character appeared before (0, 1, or 2+)
- Implement from scratch (no `nn.TransformerEncoder`)

**Components:**
- `TransformerLayer` - Single transformer layer
  1. Self-attention (single-head)
  2. Residual connection
  3. FFN (Linear → nonlinearity → Linear)
  4. Final residual connection
- `Transformer` - Full model with positional encodings
- `train_classifier` - Training loop

**Target:** >95% accuracy (reference: 98%+ in 5-10 epochs, ~20s each)

**Key Implementation Details:**
- Use Q, K, V matrices for attention
- Apply causal masking (backward-only attention)
- Include positional encodings
- Make predictions at all positions simultaneously

#### Part 2: Language Modeling (50 points)

**Task:** Character-level language model on text8
- Predict next character at each position
- **Target:** Perplexity ≤7
- **Runtime:** <10 minutes

**Requirements:**
- Properly normalized probability distributions
- Causal masking (prevent attending to future tokens)
- Can use `nn.TransformerEncoder` for this part

### Running the Code
```bash
cd "Assignment 3/a3-distrib"

# Part 1: Letter counting (backward only)
python letter_counting.py

# Letter counting (bidirectional)
python letter_counting.py --task BEFOREAFTER

# Part 2: Language modeling
python lm.py --model NEURAL
```

### Files
- `transformer.py` - Implement Transformer components
- `transformer_lm.py` - Language modeling implementation
- `letter_counting.py` - Part 1 driver
- `lm.py` - Part 2 driver
- `utils.py` - Utilities

---

## Assignment 4: Fact-checking ChatGPT Outputs

**Location:** `Assignment 4/a4-distrib/`

**Objective:** Verify factual accuracy of ChatGPT-generated biographies using Wikipedia.

### Dataset
- **Source:** FActScore (Min et al., 2023)
- ChatGPT-generated biographies with human-annotated facts
- Retrieved Wikipedia passages (BM25)
- Labels: S (supported), NS (not supported)

### Implementation Tasks

#### Part 1: Word Overlap (40 points)
- Bag-of-words overlap between fact and passages
- Compute similarity scores (cosine, Jaccard, ROUGE, etc.)
- **Target:** ≥75% accuracy

**Design Decisions:**
- Tokenization strategy
- Stemming/lemmatization
- Stopword removal
- Similarity metric
- Classification threshold

#### Part 2: Textual Entailment (40 points)
- Use pre-trained DeBERTa-v3 model
- Fine-tuned on MNLI, FEVER, ANLI
- 3-way classification: entailment/neutral/contradiction
- **Target:** ≥83% accuracy

**Implementation:**
- Split passages into sentences
- Compare fact against each sentence
- Aggregate results (max strategy)
- Optimize with word overlap pruning

#### Part 3: Error Analysis (20 points)

**Written submission** analyzing model errors:
- Examine 10 false positives + 10 false negatives
- Define 2-4 fine-grained error categories
- Provide statistics and 3 detailed examples

### Running the Code
```bash
cd "Assignment 4/a4-distrib"

# IMPORTANT: Use Python 3.10 and install requirements
pip install -r requirements.txt

# Word overlap baseline
python factchecking_main.py --mode word_overlap

# Entailment model (with GPU)
python factchecking_main.py --mode entailment --cuda

# Entailment model (CPU only)
python factchecking_main.py --mode entailment
```

### Files
- `factchecking_main.py` - Main driver (do not modify)
- `factcheck.py` - Implement fact checkers here
- `data/` - Dataset files

### Performance Considerations
- Runtime: ~10 minutes target
- Use pruning to reduce entailment calls
- Manage memory (del unused variables, gc.collect())
- Sentence splitting and cleaning required

---

## Final Project: NLI and QA with ELECTRA

**Location:** `Final Project/fp-dataset-artifacts/`

**Objective:** Fine-tune ELECTRA models on Natural Language Inference and Question Answering tasks.

### Tasks

#### Natural Language Inference (NLI)
- **Dataset:** SNLI
- **Model:** ELECTRA-small
- **Target:** ~89% accuracy (3 epochs)

#### Question Answering (QA)
- **Dataset:** SQuAD
- **Model:** ELECTRA-small
- **Target:** ~78 EM, ~86 F1 (3 epochs)

### Setup
```bash
cd "Final Project/fp-dataset-artifacts"

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
# Train NLI model
python3 run.py --do_train --task nli --dataset snli --output_dir ./trained_model/

# Train QA model
python3 run.py --do_train --task qa --dataset squad --output_dir ./trained_model/

# CPU only (no GPU)
python3 run.py --do_train --task nli --dataset snli --output_dir ./trained_model/ --no_cuda
```

### Evaluation
```bash
# Evaluate NLI model
python3 run.py --do_eval --task nli --dataset snli --model ./trained_model/ --output_dir ./eval_output/

# Evaluate QA model
python3 run.py --do_eval --task qa --dataset squad --model ./trained_model/ --output_dir ./eval_output/
```

### Files
- `run.py` - Main training/evaluation script
- `helpers.py` - Helper functions
- `requirements.txt` - Dependencies
- `README.md` - Detailed project documentation

### Data & Models
- Automatically downloaded from HuggingFace Hub
- Cached in `~/.cache/huggingface/`
- Change cache: set `HF_HOME` or `TRANSFORMERS_CACHE` environment variable

---

## Course Information

**Program:** UT Austin MS in Artificial Intelligence
**Course:** Natural Language Processing

### Key Concepts Covered
1. **Classical ML for NLP:** Perceptron, logistic regression, feature engineering
2. **Neural Networks:** Deep averaging networks, optimization techniques
3. **Word Embeddings:** GloVe, prefix embeddings, handling OOV words
4. **Transformers:** Self-attention, positional encodings, language modeling
5. **Large Language Models:** Fact-checking, textual entailment, error analysis
6. **Pre-trained Models:** ELECTRA, fine-tuning for downstream tasks

### Important Notes

#### Academic Honesty
- Discussion allowed, but **all submitted work must be your own**
- Do not share code or solutions
- Cite any external resources used

#### Submission Guidelines
- **Assignment 1:** Submit `models.py` only
- **Assignment 2:** Submit `optimization.py` and `models.py`
- **Assignment 3:** Submit `transformer.py` and `transformer_lm.py`
- **Assignment 4:** Submit `factcheck.py` + written analysis

#### General Tips
1. **Start early** - debugging neural networks takes time
2. **Use small experiments** - tune on small datasets before scaling up
3. **Monitor training** - print losses, check dev accuracy frequently
4. **Manage resources** - respect runtime and memory constraints
5. **Test locally** - ensure code runs before autograder submission

### References

**Assignment 1:**
- Socher et al. (2013) - Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank

**Assignment 2:**
- Iyyer et al. (2015) - Deep Unordered Composition Rivals Syntactic Methods for Text Classification
- Pennington et al. (2014) - GloVe: Global Vectors for Word Representation

**Assignment 3:**
- Mikolov et al. (2012) - Subword Language Modeling with Neural Networks
- Vaswani et al. (2017) - Attention Is All You Need

**Assignment 4:**
- Min et al. (2023) - FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation
- He et al. (2020) - DeBERTa: Decoding-enhanced BERT with Disentangled Attention
- Williams et al. (2018) - MNLI: A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference
- Thorne et al. (2018) - FEVER: a large-scale dataset for fact extraction and VERification

---

## Repository Structure

```
NLP/
├── Assignment 0/                # Introduction to tokenization
│   ├── tokenizer.py
│   ├── count.py
│   └── token_counter.py
│
├── Assignment 1/                # Sentiment classification (classical ML)
│   └── a1-distrib/
│       ├── a1.pdf              # Assignment instructions
│       ├── models.py           # Your implementation
│       ├── sentiment_classifier.py
│       └── data/
│
├── Assignment 2/                # Neural networks & word embeddings
│   └── a2-distrib/
│       ├── a2.pdf              # Assignment instructions
│       ├── models.py           # Your implementation
│       ├── optimization.py     # Your implementation
│       ├── neural_sentiment_classifier.py
│       └── data/
│
├── Assignment 3/                # Transformer language modeling
│   └── a3-distrib/
│       ├── a3.pdf              # Assignment instructions
│       ├── transformer.py      # Your implementation
│       ├── transformer_lm.py   # Your implementation
│       ├── letter_counting.py
│       ├── lm.py
│       └── data/
│
├── Assignment 4/                # Fact-checking with LLMs
│   └── a4-distrib/
│       ├── a4.pdf              # Assignment instructions
│       ├── factcheck.py        # Your implementation
│       ├── factchecking_main.py
│       └── data/
│
├── Final Project/               # NLI and QA with ELECTRA
│   └── fp-dataset-artifacts/
│       ├── README.md           # Project documentation
│       ├── run.py
│       ├── helpers.py
│       └── requirements.txt
│
├── Module 2/                    # Example code
│   └── ffnn_example.py         # FFNN XOR example
│
└── README.md                    # This file
```

---

## License

This repository contains coursework for educational purposes as part of the UT Austin MS AI program.

