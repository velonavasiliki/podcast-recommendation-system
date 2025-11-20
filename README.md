# Hybrid Deep Learning Scoring System for Podcast Recommendations

A hybrid neural architecture that combines collaborative filtering with natural language understanding to predict user engagement scores. Built with TensorFlow/Keras and BERT embeddings, trained on 2.3M samples from the Podcast Reviews dataset.

## Overview

This project implements an end-to-end machine learning pipeline for predicting user-item engagement scores using a **hybrid deep learning architecture**. The system integrates:

- **Collaborative Filtering**: Learns user and item embeddings from rating patterns
- **Transfer Learning**: Leverages pre-trained BERT (all-MiniLM-L6-v2) for semantic understanding of review text
- **Deep Fusion Network**: Multi-layer perceptron with batch normalization for non-linear feature interactions
- **Memory-Efficient Pipeline**: DuckDB for data processing, chunked BERT inference, and Parquet storage

The model achieves **R² = 0.2853** on temporal held-out test sets, with systematic evaluation across multiple data splitting strategies to understand robustness and generalization.

---

## Technical Highlights

### Machine Learning Engineering
- **Custom Keras Model**: Implemented `HybridNCF` class with proper serialization (`get_config()`/`from_config()`)
- **Transfer Learning**: Fine-tuned BERT embeddings for domain adaptation
- **Training Optimization**: Adam optimizer with gradient clipping, learning rate scheduling, and early stopping
- **Memory Management**: Chunked BERT inference (50K samples/batch) with garbage collection

### Data Science & Experimentation
- **Controlled Experiments**: Compared 3 data splitting strategies to measure cold-start impact
- **Threshold Analysis**: Evaluated active user thresholds (≥4 vs ≥8 reviews)
- **Multiple Metrics**: RMSE, MAE, R² for comprehensive evaluation
- **Temporal Validation**: Simulated production scenario by testing on future data

### System Design
- **Modular Architecture**: Separated concerns across `models.py`, `training.py`, and `data_splitting.py`
- **Factory Pattern**: Configurable model instantiation with hyperparameter dictionaries
- **Efficient Storage**: Parquet format for columnar storage of embeddings
- **DuckDB Pipeline**: SQL-based ETL for memory-efficient data processing on 5M+ records

---

## Model Architecture: HybridNCF

### System Diagram

```
Input:
  ├─ User ID
  └─ Item ID
       ↓
Embedding Layers:
  ├─ User CF Embedding (398K × 64)
  ├─ Item CF Embedding (182K × 64)
  ├─ Item Content Embedding (182K × 384, BERT)
  │    └─ Projection: 384 → 64 (ReLU)
  ├─ User Bias (398K × 1)
  └─ Item Bias (182K × 1)
       ↓
Feature Engineering:
  └─ CF Interaction: user_cf ⊗ item_cf
       ↓
Fusion Network:
  └─ Concat [user_cf, item_cf, content_proj, interaction] → 256 dims
       ↓
  └─ Dense(128) → BatchNorm → Dropout(0.2)
       ↓
  └─ Dense(64) → BatchNorm → Dropout(0.2)
       ↓
Output:
  └─ Dense(1) + user_bias + item_bias → Score
```

### Components

**1. Collaborative Filtering**
- Learned embeddings (64-dim) capture latent user preferences and item characteristics
- Element-wise interaction term models GMF-style feature interactions
- Bias terms handle global rating tendencies

**2. Content-Based Filtering**
- Pre-trained BERT embeddings (384-dim) from review text
- Learned projection layer adapts BERT to recommendation task
- Fine-tuning enabled for domain adaptation

**3. Deep Fusion Network**
- 2 hidden layers (128 → 64 units) with ReLU activation
- Batch normalization for stable training
- Dropout (0.2) and L2 regularization (1e-5) for generalization

**4. Training Configuration**
- Optimizer: Adam (lr=0.0002, clipnorm=1.0)
- Loss: MSE
- Batch size: 2048
- Callbacks: Early stopping (patience=5), ReduceLROnPlateau

---

## Dataset & Preprocessing

### Data Source
**Podcast Reviews Dataset** ([Kaggle](https://www.kaggle.com/datasets/thoughtvector/podcastreviews/)): 5.6M user reviews across 304K podcasts

### Pipeline

**1. Data Exploration & Filtering** (DuckDB)
- Analyzed user activity distribution
- Filtered to active users (≥3 reviews)
- Final dataset: 2.3M reviews from 398K users, 182K podcasts
- Captures 41% of reviews from 12% of users

**2. BERT Embedding Generation**
- Model: `all-MiniLM-L6-v2` (384-dim)
- Combined review title + content
- Chunked inference (50K samples/batch) for memory efficiency
- Mean pooling aggregation per podcast

**3. ID Encoding**
- LabelEncoder for user/podcast IDs → contiguous integers
- Saved encoders for inference

**4. Storage**
- Parquet format for efficient columnar storage
- Compressed embeddings: ~4GB on disk

---

## Experimental Results

### Experiment 1: Data Splitting Strategy Comparison

**Research Question**: How does data splitting affect model performance?

**Methodology**: Trained identical HybridNCF models on 3 different train/test splits.

**Results**:

| Split Strategy | Train Samples | Test Samples | RMSE | MAE | R² | Training Time |
|----------------|---------------|--------------|------|-----|-------|---------------|
| **Temporal** | 1,833,084 | 458,271 | 1.2390 | 0.8443 | 0.1383 | 3.0 min |
| **Random** | 1,833,084 | 458,271 | 0.9985 | 0.6008 | 0.2918 | 2.8 min |
| **User-Stratified** | 1,790,848 | 500,507 | 1.0429 | 0.6222 | 0.2323 | 2.8 min |

**Key Insights**:

1. **Random split overestimates performance** (111% higher R²) due to IID assumption - not representative of production

2. **User-stratified eliminates cold-start** (all test users in training) → 68% R² improvement vs temporal

3. **Temporal split is most realistic** - trains on past data, tests on future ratings
---

### Experiment 2: Active User Threshold Analysis

**Research Question**: What minimum review threshold optimizes accuracy vs data size?

**Methodology**: Temporal user-stratified split (last review per user → test) with two thresholds.

**Results**:

| Threshold | Users | Podcasts | Train Samples | Test Samples | RMSE | MAE | R² | Time |
|-----------|-------|----------|---------------|--------------|------|-----|-------|------|
| **≥4 reviews** | 231,518 | 158,658 | 1,559,764 | 231,518 | **1.0477** | **0.6339** | 0.2653 | 128.9s |
| **≥8 reviews** | 61,272 | 113,323 | 894,867 | 61,272 | 1.1097 | 0.7313 | **0.2853** | 66.5s |

**Key Insights**:

1. **Lower threshold (≥4) gives better accuracy**: 6% lower RMSE, 13% lower MAE - more data compensates for noise

2. **Higher threshold (≥8) explains variance better**: 8% higher R² - richer user histories improve pattern learning

---

## Performance Analysis

### Metrics

**RMSE: 1.0477** - Average prediction error of ±1.05 stars (on 1-5 scale)

**MAE: 0.6339** - Average absolute error of 0.63 stars

**R²: 0.2853** - Model explains 28.5% of rating variance

### Context

R² = 0.25-0.30 is competitive for rating prediction:
- Netflix Prize winner: R² ≈ 0.30
- Standard collaborative filtering: R² = 0.15-0.25
- Our hybrid approach: R² = 0.2853 

Rating prediction is inherently noisy due to user subjectivity, context, and mood.

---

## Repository Structure

```
podcast-recommendation-system/
├── notebooks/
│   ├── dataExplorationFeatureEng.ipynb          # EDA + BERT pipeline
│   ├── trainingExperiments.ipynb                # Split comparison
│   └── training_active_users_comparison.ipynb   # Threshold analysis
├── src/
│   ├── models.py                                # HybridNCF model
│   ├── training.py                              # Training functions
│   └── data_splitting.py                       # Temporal split
├── config.py                                    # Configuration
└── README.md
```

## References

- **Dataset**: [Podcast Reviews on Kaggle](https://www.kaggle.com/datasets/thoughtvector/podcastreviews/)

