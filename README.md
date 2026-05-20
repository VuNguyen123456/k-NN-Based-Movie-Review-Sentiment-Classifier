# ReviewSense: k-NN Based Movie Review Sentiment Classifier

## Overview

ReviewSense is a sentiment classification system designed to analyze free-form movie reviews and predict whether they express a positive or negative sentiment. The project explores how traditional machine learning techniques—without deep learning—can still perform effectively on large-scale text classification tasks.

At its core, the system implements a fully custom k-Nearest Neighbors (k-NN) classifier from scratch, including feature engineering, similarity computation, and evaluation logic.

---

## Problem Statement

Given a dataset of 25,000 labeled movie reviews and 10,000 unlabeled reviews, the goal is to predict sentiment polarity:

- **+1 → Positive sentiment**
- **-1 → Negative sentiment**

Each review is raw text ending with a special `#EOF` token, requiring preprocessing before modeling.

---

## Key Features

### Custom k-NN Implementation
- Built entirely from scratch (no ML library classifier used)
- Manual implementation of:
  - Distance/similarity computation
  - Neighbor selection
  - Voting-based prediction

### Text Processing Pipeline
- Tokenization and normalization
- Stopword removal
- Optional stemming/lemmatization
- N-gram feature support

### Feature Engineering
- Bag-of-Words (binary & frequency-based)
- TF-IDF representation
- Vocabulary trimming / top-K feature selection

### Similarity Metrics
- Cosine similarity (primary)
- Euclidean distance (baseline)
- Weighted neighbor voting (experimental)

---

## System Pipeline

1. **Data Loading**
   - Parse labeled training data and unlabeled test data
   - Handle formatting and EOF markers

2. **Preprocessing**
   - Clean text
   - Tokenize reviews
   - Normalize and filter tokens

3. **Feature Extraction**
   - Build vocabulary
   - Convert text into vector representations

4. **Similarity Computation**
   - Compute similarity between test and training samples

5. **k-NN Classification**
   - Select K nearest neighbors
   - Predict via majority (or weighted) voting

6. **Output Generation**
   - Export predictions in submission format

---

## Experiments

Several configurations were tested to optimize performance:

### Feature Representations
- TF-IDF consistently outperformed raw counts
- Binary bag-of-words was fast but less accurate

### Hyperparameter Tuning
- Tested multiple values of K
- Smaller K → more sensitive predictions
- Larger K → smoother, more stable predictions

### Similarity Functions
- Cosine similarity performed best for sparse text data
- Euclidean distance was less effective in high-dimensional space

---

## Performance Optimizations

Because k-NN is computationally expensive, several optimizations were implemented:

- Sparse matrix representations
- Precomputed feature vectors
- Vocabulary reduction to improve runtime
- Vectorized similarity calculations

These improvements significantly reduced inference time while maintaining accuracy.

---

## Tech Stack

- Python
- NumPy
- SciPy (sparse matrix support)
- scikit-learn (utilities only, not classifier)
- Jupyter Notebook

---

## Key Insights

This project highlights how much impact feature engineering has in classical NLP. While k-NN is simple in concept, its performance heavily depends on:

- Representation quality (TF-IDF vs raw counts)
- Distance metric choice
- Dimensionality control
- Proper tuning of K

In many cases, feature design mattered more than the classifier itself.

---

## Future Improvements

- Faster approximate nearest neighbor search (e.g., FAISS)
- Dimensionality reduction (SVD / PCA for text)
- GPU-accelerated similarity computation
- Ensemble with other classical models (Naive Bayes, Logistic Regression)

---
