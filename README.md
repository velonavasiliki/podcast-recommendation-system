# Podcast Recommendation System

A hybrid neural collaborative filtering system for podcast recommendations that combines collaborative filtering with content-based filtering using BERT embeddings.

## Overview

This project implements a deep learning recommendation system that predicts podcast ratings by combining:
- **Collaborative Filtering**: User-item interaction patterns learned through embeddings
- **Content-Based Filtering**: Semantic understanding from BERT-encoded review text
- **Deep Neural Networks**: Multi-layer perceptron fusion layers for learning complex feature interactions

The system achieves strong performance on the Podcast Reviews dataset, with comprehensive evaluation across different data splitting strategies.

## Features

- Hybrid recommendation architecture combining collaborative and content-based signals
- Transfer learning using pre-trained BERT embeddings (all-MiniLM-L6-v2)
- Extensive data preprocessing and feature engineering pipeline
- Multiple evaluation strategies (temporal, random, and user-stratified splits)
- Cold-start analysis and mitigation strategies
- Comprehensive Jupyter notebooks for reproducibility

## Dataset

This project uses the **Podcast Reviews** dataset from Kaggle:
- **Source**: [Podcast Reviews Dataset](https://www.kaggle.com/datasets/thoughtvector/podcastreviews/)
- **File**: `reviews.json`
- **Size**: 5M+ reviews
- **After preprocessing**: ~2.9M reviews from active users (3+ reviews)

* Download `reviews.json` from the [Kaggle dataset page](https://www.kaggle.com/datasets/thoughtvector/podcastreviews/)

 
## Project Structure

```
podcast-recommendation-system/
├── notebooks/
│   ├── dataExplorationFeatureEng.ipynb  # Data preprocessing and feature engineering
│   └── trainingExperiments.ipynb        # Model training and evaluation
├── src/
│   └── models.py                         # HybridNCF model architecture
├── data/                                 # Dataset directory (not tracked)
│   └── reviews.json                      # Raw dataset (download separately)
├── .gitignore
└── README.md
```

## Usage

### 1. Data Preprocessing

Run the `dataExplorationFeatureEng.ipynb` notebook to:
- Load and explore the raw dataset
- Filter for active users (3+ reviews)
- Generate BERT embeddings from review text
- Create train/test splits
- Save processed data in Parquet format

### 2. Model Training

Run the `trainingExperiments.ipynb` notebook to:
- Train the Hybrid NCF model on three different data splits:
  - Temporal split (80-20%)
  - Random split (80-20%)
  - User-stratified split (80-20%)
- Evaluate model performance (RMSE, MAE, R²)
- Analyze cold-start problem impact
- Generate performance comparison visualizations

### 3. Model Architecture

The Hybrid NCF model combines:
- **User embeddings**: 64-dimensional learned representations
- **Item embeddings**: 64-dimensional learned representations
- **BERT embeddings**: 384-dimensional pre-trained embeddings projected to 64D
- **Fusion network**: Two MLP layers (128→64 units) with batch normalization, dropout, and L2 regularization
- **Training**: Adam optimizer with learning rate scheduling and early stopping

## Results

Performance comparison across splitting strategies:

| Split Strategy    | RMSE   | MAE    | R²     |
|-------------------|--------|--------|--------|
| Temporal         | 1.2215 | 0.7926 | 0.1624 |
| Random           | 1.0004 | 0.6112 | 0.2891 |
| User-Stratified  | 1.0035 | 0.6007 | 0.2750 |

**Key Finding**: The temporal split shows lower performance due to the cold-start problem - many test users are completely new and don't appear in the training set. This represents the most realistic production scenario.

## Model Performance

- **Best RMSE**: 1.00 (Random split)
- **Production-realistic RMSE**: 1.22 (Temporal split)
- The hybrid approach effectively leverages both collaborative and content signals
- Content-based features (BERT) help mitigate cold-start issues for new users

