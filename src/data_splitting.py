"""
This module provides functions for splitting podcast review data into
train/test sets with temporal ordering and user stratification.
"""

import pandas as pd
from typing import Tuple

def temporal_user_stratified_split(
    df: pd.DataFrame,
    user_col: str = 'auth_id_encoded',
    time_col: str = 'created_at',
    active_users_k: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally with user stratification.

    For each active user (>= k reviews), reviews are sorted by time and split such that:
    - The most recent review goes to test
    - All remaining reviews go to training

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the review data
    user_col : str
        Column name for user identifiers
    time_col : str
        Column name for timestamps
    active_users_k : int
        Minimum number of reviews for a user to be considered active (default: 3)

    Returns
    -------
    train_df : pd.DataFrame
        Training set (all reviews except last 1 per user)
    test_df : pd.DataFrame
        Test set (last review per user)
    """

    # Sort by user and time
    df_sorted = df.sort_values([user_col, time_col]).reset_index(drop=True)

    # Filter to active users (with minimum reviews)
    user_counts = df_sorted[user_col].value_counts()
    active_users = user_counts[user_counts >= active_users_k].index
    df_active = df_sorted[df_sorted[user_col].isin(active_users)].copy()

    print(f"Total users: {df_sorted[user_col].nunique():,}")
    print(f"Active users (>= {active_users_k} reviews): {len(active_users):,}")
    print(f"Total reviews from active users: {len(df_active):,}")

    # Initialize index lists
    train_indices = []
    test_indices = []

    # Group by user and split temporally
    for user_id, group in df_active.groupby(user_col):
        indices = group.index.tolist()

        # Last review -> test
        # Rest -> training
        test_indices.append(indices[-1])
        train_indices.extend(indices[:-1])

    # Create split DataFrames
    train_df = df_active.loc[train_indices].reset_index(drop=True)
    test_df = df_active.loc[test_indices].reset_index(drop=True)

    # Print split statistics
    print(f"\nSplit Statistics:")
    print(f"  Training: {len(train_df):,} reviews")
    print(f"  Test:     {len(test_df):,} reviews")
    print(f"  Total:    {len(train_df) + len(test_df):,} reviews")

    # Verify each user has exactly 1 sample in test
    assert len(test_df) == len(active_users), "Test set should have 1 review per user"

    return train_df, test_df
