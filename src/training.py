"""
This module provides functions for data preparation, embedding aggregation,
and model training/evaluation.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import gc
import time
from typing import Dict, Tuple, Any
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.models import create_model


def aggregate_embeddings(
    df: pd.DataFrame,
    embedding_col: str = 'bert_embedding_all-MiniLM-L6-v2',
    podcast_col: str = 'podcast_id_encoded'
) -> np.ndarray:
    """
    Aggregate BERT embeddings by podcast using mean pooling.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing review data with embeddings
    embedding_col : str
        Column name containing the BERT embeddings
    podcast_col : str
        Column name for podcast identifiers

    Returns
    -------
    np.ndarray
        Aggregated embeddings array of shape (n_podcasts, embedding_dim)
    """
    n_podcasts = df[podcast_col].max() + 1
    embedding_dim = len(df[embedding_col].iloc[0])

    embeddings = np.zeros((n_podcasts, embedding_dim), dtype=np.float32)
    counts = np.zeros(n_podcasts, dtype=np.int32)

    # Process in chunks to manage memory
    chunk_size = 500000
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        chunk_embeddings = np.stack(chunk[embedding_col].values)
        chunk_ids = chunk[podcast_col].values

        np.add.at(embeddings, chunk_ids, chunk_embeddings)
        np.add.at(counts, chunk_ids, 1)

    # Average embeddings
    mask = counts > 0
    embeddings[mask] /= counts[mask, np.newaxis]

    return embeddings


def prepare_training_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_col: str = 'auth_id_encoded',
    item_col: str = 'podcast_id_encoded',
    rating_col: str = 'rating'
) -> Tuple[Dict, np.ndarray, Dict, np.ndarray]:
    """
    Prepare X and y arrays for model training.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data
    user_col : str
        Column name for user IDs
    item_col : str
        Column name for item IDs
    rating_col : str
        Column name for ratings
 
    Returns
    -------
    X_train : dict
        Training inputs {'user': array, 'item': array}
    y_train : np.ndarray
        Training targets
    X_test : dict
        Test inputs
    y_test : np.ndarray
        Test targets
    """
    X_train = {
        'user': train_df[user_col].values.astype(np.int32),
        'item': train_df[item_col].values.astype(np.int32)
    }
    y_train = train_df[rating_col].values.astype(np.float32)

    X_test = {
        'user': test_df[user_col].values.astype(np.int32),
        'item': test_df[item_col].values.astype(np.int32)
    }
    y_test = test_df[rating_col].values.astype(np.float32)

    return X_train, y_train, X_test, y_test


def encode_ids(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_col: str = 'author_id',
    podcast_col: str = 'podcast_id'
) -> Tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """
    Re-encode user and podcast IDs for a specific data subset.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data
    user_col : str
        Column name for original user IDs
    podcast_col : str
        Column name for original podcast IDs

    Returns
    -------
    train_df : pd.DataFrame
        Training data with encoded columns
    test_df : pd.DataFrame
        Test data with encoded columns
    n_users : int
        Number of unique users
    n_podcasts : int
        Number of unique podcasts
    """
    user_encoder = LabelEncoder()
    podcast_encoder = LabelEncoder()

    all_users = pd.concat([train_df[user_col], test_df[user_col]]).unique()
    all_podcasts = pd.concat([train_df[podcast_col], test_df[podcast_col]]).unique()

    user_encoder.fit(all_users)
    podcast_encoder.fit(all_podcasts)

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df['auth_id_encoded'] = user_encoder.transform(train_df[user_col])
    train_df['podcast_id_encoded'] = podcast_encoder.transform(train_df[podcast_col])
    test_df['auth_id_encoded'] = user_encoder.transform(test_df[user_col])
    test_df['podcast_id_encoded'] = podcast_encoder.transform(test_df[podcast_col])

    return train_df, test_df, len(all_users), len(all_podcasts)


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_config: Dict[str, Any],
    regime_name: str,
    user_col: str = 'author_id',
    podcast_col: str = 'podcast_id',
    batch_size: int = 2048,
    epochs: int = 30,
    validation_split: float = 0.15
) -> Dict[str, Any]:
    """
    Train model and return evaluation metrics.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data
    model_config : dict
        Model hyperparameters
    regime_name : str
        Name for this training regime (for logging)
    user_col : str
        Column name for user IDs
    podcast_col : str
        Column name for podcast IDs
    batch_size : int
        Training batch size
    epochs : int
        Maximum training epochs
    validation_split : float
        Fraction of training data for validation

    Returns
    -------
    dict
        Results including metrics, history, and metadata
    """
    print(f"\n{'='*60}")
    print(f"Training: {regime_name}")
    print(f"{'='*60}")

    # Re-encode IDs for this subset
    train_df, test_df, n_users, n_podcasts = encode_ids(
        train_df, test_df, user_col, podcast_col
    )

    print(f"Users: {n_users:,}")
    print(f"Podcasts: {n_podcasts:,}")
    print(f"Train samples: {len(train_df):,}")
    print(f"Test samples: {len(test_df):,}")

    # Aggregate embeddings from training set
    print("\nAggregating embeddings...")
    pretrained_embeddings = aggregate_embeddings(train_df)

    # Prepare data
    X_train, y_train, X_test, y_test = prepare_training_data(train_df, test_df)

    # Clear session and create model
    tf.keras.backend.clear_session()
    gc.collect()

    model = create_model(n_users, n_podcasts, pretrained_embeddings, model_config)

    optimizer = Adam(
        learning_rate=model_config.get('learning_rate', 0.0002),
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            min_delta=0.0001,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train
    print("\nTraining...")
    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.time() - start_time

    # Evaluate
    predictions = model.predict(X_test, batch_size=batch_size, verbose=0).flatten()

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\n{regime_name} Results:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  Training time: {training_time:.1f}s")

    return {
        'regime': regime_name,
        'n_users': n_users,
        'n_podcasts': n_podcasts,
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'training_time': training_time,
        'history': history.history,
        'model': model
    }
