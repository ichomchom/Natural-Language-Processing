# Natural Language Processing Portfolio

A comprehensive collection of NLP implementations from the UT Austin MS AI program, demonstrating progression from classical machine learning to modern transformer-based architectures.

## Table of Contents
- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Projects](#projects)
  - [1. Sentiment Classification with Classical ML](#1-sentiment-classification-with-classical-ml)
  - [2. Neural Networks & Word Embeddings](#2-neural-networks--word-embeddings)
  - [3. Transformer Language Modeling](#3-transformer-language-modeling)
  - [4. Fact-Checking LLM Outputs](#4-fact-checking-llm-outputs)
  - [5. Fine-tuning ELECTRA for NLI and QA](#5-fine-tuning-electra-for-nli-and-qa)
- [Key Skills Demonstrated](#key-skills-demonstrated)
- [References](#references)

---

## Overview

This repository showcases implementations of core NLP techniques, from foundational machine learning algorithms to state-of-the-art transformer models. Each project builds upon previous concepts while introducing new architectures and methodologies.

**Technologies:** Python, PyTorch, NumPy, HuggingFace Transformers, NLTK, spaCy

---

## Environment Setup

### Requirements
- Python 3.10+
- PyTorch
- NumPy, nltk, spacy
- HuggingFace Transformers

### Installation
```bash
# Create virtual environment
conda create -n nlp_env python=3.10
conda activate nlp_env

# Install PyTorch (visit https://pytorch.org for system-specific instructions)
pip install torch

# Install dependencies
pip install numpy nltk spacy transformers
```

---

## Projects

### 1. Sentiment Classification with Classical ML

**Location:** `Assignment 1/a1-distrib/`

Implements classical machine learning approaches for binary sentiment classification on movie reviews from Rotten Tomatoes.

#### Implementations

**Perceptron Classifier**
- Custom perceptron with bag-of-words unigram features
- Sparse vector representations for efficiency
- Random shuffling and learning rate scheduling
- **Performance:** 74%+ accuracy on movie review sentiment

**Logistic Regression**
- Gradient-based optimization
- Negative log-likelihood loss
- Achieved 77%+ accuracy

**Feature Engineering**
- Unigram features with various preprocessing strategies
- Bigram features for capturing local context
- Advanced feature extractors with tf-idf weighting, stopword filtering
- Experimentation with tokenization, stemming, and feature value schemes

#### Usage
```bash
cd "Assignment 1/a1-distrib"

# Perceptron classifier
python sentiment_classifier.py --model PERCEPTRON --feats UNIGRAM

# Logistic regression with different feature sets
python sentiment_classifier.py --model LR --feats UNIGRAM
python sentiment_classifier.py --model LR --feats BIGRAM
python sentiment_classifier.py --model LR --feats BETTER
```

#### Technical Highlights
- Efficient sparse vector representations using Python Counter
- Custom Indexer class for feature-to-index mapping
- Systematic comparison of different feature extraction strategies
- Understanding of bias-variance tradeoffs in feature design

---

### 2. Neural Networks & Word Embeddings

**Location:** `Assignment 2/a2-distrib/`

Explores deep learning for sentiment analysis using pre-trained word embeddings and feedforward neural networks.

#### Implementations

**Optimization Fundamentals**
- Manual implementation of gradient descent
- Analysis of step size effects on convergence
- Optimal learning rate selection

**Deep Averaging Network (DAN)**
- Feedforward architecture with averaged word embeddings
- Pre-trained GloVe embeddings (50d and 300d)
- Mini-batch training for efficiency
- Dropout regularization
- **Performance:** 77%+ accuracy on sentiment classification

**Robustness to Typos**
- Handling misspellings in test data through:
  - Spelling correction using edit distance
  - Prefix embeddings (first 3 characters)
  - Character-level representations
- **Performance:** 74%+ accuracy on corrupted text

#### Usage
```bash
cd "Assignment 2/a2-distrib"

# Optimization experiments
python optimization.py

# Train deep averaging network
python neural_sentiment_classifier.py

# Test robustness to typos
python neural_sentiment_classifier.py --use_typo_setting

# Fast debugging with 50d embeddings
python neural_sentiment_classifier.py --word_vecs_path data/glove.6B.50d-relativized.txt
```

#### Technical Highlights
- PyTorch model implementation with custom `nn.Module` subclasses
- Batching with padding for variable-length sequences
- Integration of pre-trained word embeddings
- Strategies for handling out-of-vocabulary words
- Analysis of generalization to different data distributions

---

### 3. Transformer Language Modeling

**Location:** `Assignment 3/a3-distrib/`

Implements transformer architecture from scratch for sequence modeling tasks.

#### Implementations

**Custom Transformer Encoder**
- Built from scratch without using `nn.TransformerEncoder`
- Self-attention mechanism with Q, K, V projections
- Positional encodings for sequence order
- Residual connections and feedforward layers
- Causal masking for autoregressive modeling

**Character Counting Task**
- Predicts occurrence counts of characters in context
- Demonstrates transformer's ability to aggregate information across positions
- **Performance:** 95%+ accuracy, demonstrating effective attention patterns

**Character-Level Language Model**
- Trained on text8 dataset (100M Wikipedia characters)
- Predicts next character at each position
- Proper probability normalization
- **Performance:** Perplexity of 6.3 on held-out data

#### Usage
```bash
cd "Assignment 3/a3-distrib"

# Character counting with transformers
python letter_counting.py
python letter_counting.py --task BEFOREAFTER

# Language modeling
python lm.py --model NEURAL
```

#### Technical Highlights
- Ground-up implementation of self-attention mechanisms
- Understanding of attention masks and causality
- Positional encoding strategies
- Sequence-to-sequence prediction architecture
- Perplexity evaluation for language models
- Visualization of learned attention patterns

---

### 4. Fact-Checking LLM Outputs

**Location:** `Assignment 4/a4-distrib/`

Develops systems to verify factual claims from ChatGPT-generated biographies against Wikipedia sources.

#### Implementations

**Word Overlap Methods**
- Multiple similarity metrics: cosine similarity, Jaccard, ROUGE
- Tf-idf vectorization
- Tokenization and preprocessing pipelines
- Threshold optimization for classification
- **Performance:** 75%+ accuracy

**Neural Textual Entailment**
- Pre-trained DeBERTa-v3 model (fine-tuned on MNLI, FEVER, ANLI)
- Three-way classification: entailment, neutral, contradiction
- Sentence-level fact verification
- Passage retrieval and aggregation strategies
- Optimization through pruning low-overlap examples
- **Performance:** 83%+ accuracy

**Error Analysis**
- Systematic categorization of false positives and false negatives
- Fine-grained error taxonomy
- Insights into model limitations and failure modes

#### Usage
```bash
cd "Assignment 4/a4-distrib"

# Word overlap baseline
python factchecking_main.py --mode word_overlap

# Neural entailment model
python factchecking_main.py --mode entailment

# With GPU acceleration
python factchecking_main.py --mode entailment --cuda
```

#### Technical Highlights
- Integration of pre-trained language models
- Text preprocessing and sentence segmentation
- Multi-stage pipeline design (retrieval → entailment)
- Performance optimization and memory management
- Practical application to AI safety and trustworthiness

---

### 5. Fine-tuning ELECTRA for NLI and QA

**Location:** `Final Project/fp-dataset-artifacts/`

Fine-tunes ELECTRA transformer models for Natural Language Inference and extractive Question Answering.

#### Implementations

**Natural Language Inference (SNLI)**
- Fine-tuned ELECTRA-small on Stanford Natural Language Inference dataset
- Three-way classification: entailment, neutral, contradiction
- **Performance:** ~89% accuracy

**Question Answering (SQuAD)**
- Fine-tuned ELECTRA-small on Stanford Question Answering Dataset
- Extractive QA with span prediction
- **Performance:** ~78 EM score, ~86 F1 score

#### Usage
```bash
cd "Final Project/fp-dataset-artifacts"

# Train NLI model
python3 run.py --do_train --task nli --dataset snli --output_dir ./trained_model/

# Train QA model
python3 run.py --do_train --task qa --dataset squad --output_dir ./trained_model/

# Evaluate models
python3 run.py --do_eval --task nli --dataset snli \
    --model ./trained_model/ --output_dir ./eval_output/
```

#### Technical Highlights
- HuggingFace Transformers integration
- Transfer learning from pre-trained models
- Dataset processing with HuggingFace Datasets
- Training loop customization
- Evaluation metrics for different tasks (accuracy, EM, F1)

---

## Key Skills Demonstrated

### Machine Learning Fundamentals
- Feature engineering and extraction
- Classical ML algorithms (Perceptron, Logistic Regression)
- Gradient descent optimization
- Hyperparameter tuning
- Cross-validation and model evaluation

### Deep Learning
- PyTorch model implementation
- Custom neural network architectures
- Backpropagation and gradient computation
- Batching and data preprocessing
- Regularization techniques (dropout)

### NLP-Specific Techniques
- Tokenization and text preprocessing
- Word embeddings (GloVe)
- Sequence modeling
- Attention mechanisms
- Language modeling and perplexity
- Textual entailment

### Transformer Architecture
- Self-attention implementation
- Positional encodings
- Multi-head attention
- Residual connections
- Layer normalization
- Causal masking

### Pre-trained Models
- Transfer learning
- Fine-tuning strategies
- HuggingFace ecosystem
- Model evaluation and analysis

### Software Engineering
- Efficient sparse representations
- Memory optimization
- Runtime performance tuning
- Modular code design
- Experiment management

---

## Repository Structure

```
NLP/
├── Assignment 0/                # Text processing and tokenization
│   ├── tokenizer.py
│   ├── count.py
│   └── token_counter.py
│
├── Assignment 1/                # Sentiment classification (classical ML)
│   └── a1-distrib/
│       ├── models.py           # Perceptron, LR, feature extractors
│       ├── sentiment_classifier.py
│       └── data/
│
├── Assignment 2/                # Neural networks & word embeddings
│   └── a2-distrib/
│       ├── models.py           # Deep averaging network
│       ├── optimization.py     # Gradient descent implementation
│       ├── neural_sentiment_classifier.py
│       └── data/               # GloVe embeddings
│
├── Assignment 3/                # Transformer language modeling
│   └── a3-distrib/
│       ├── transformer.py      # Transformer from scratch
│       ├── transformer_lm.py   # Language model implementation
│       ├── letter_counting.py
│       └── lm.py
│
├── Assignment 4/                # Fact-checking with LLMs
│   └── a4-distrib/
│       ├── factcheck.py        # Entailment & word overlap methods
│       ├── factchecking_main.py
│       └── data/               # FActScore dataset
│
├── Final Project/               # NLI and QA with ELECTRA
│   └── fp-dataset-artifacts/
│       ├── run.py              # Training and evaluation
│       ├── helpers.py
│       └── requirements.txt
│
└── README.md
```

---

## References

### Datasets & Tasks
- **Sentiment Analysis:** Socher et al. (2013) - Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank
- **NLI:** Williams et al. (2018) - MNLI: A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference
- **Fact Verification:** Min et al. (2023) - FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation
- **Fact Extraction:** Thorne et al. (2018) - FEVER: a large-scale dataset for fact extraction and VERification

### Models & Methods
- **Word Embeddings:** Pennington et al. (2014) - GloVe: Global Vectors for Word Representation
- **Deep Averaging Networks:** Iyyer et al. (2015) - Deep Unordered Composition Rivals Syntactic Methods for Text Classification
- **Transformers:** Vaswani et al. (2017) - Attention Is All You Need
- **DeBERTa:** He et al. (2020) - DeBERTa: Decoding-enhanced BERT with Disentangled Attention
- **Language Modeling:** Mikolov et al. (2012) - Subword Language Modeling with Neural Networks

### Applications
- **Summarization Consistency:** Laban et al. (2022) - SummaC: Re-visiting NLI-based models for inconsistency detection in summarization
- **Adversarial NLI:** Nie et al. (2020) - Adversarial NLI: A new benchmark for natural language understanding

---

**Program:** UT Austin MS in Artificial Intelligence
**Course:** Natural Language Processing

