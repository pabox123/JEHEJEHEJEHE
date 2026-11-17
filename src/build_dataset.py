"""
Dataset builder script for GoEmotions.
Loads raw data, applies filtering, balancing, preprocessing, and saves processed splits.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import (
    load_and_prepare_goemotions,
    apply_preprocessing
)

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
RANDOM_STATE = 42
N_SAMPLES_PER_CLASS = 500
TEST_SIZE = 0.2

def main():
    print("Starting GoEmotions dataset preparation pipeline...")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f" Loading and filtering data from {RAW_DIR}...")
    df_balanced = load_and_prepare_goemotions(
        raw_dir=RAW_DIR,
        n_samples_per_class=N_SAMPLES_PER_CLASS,
        random_state=RANDOM_STATE
    )
    print(f"Balanced dataset created: {len(df_balanced)} samples ({N_SAMPLES_PER_CLASS} positive, {N_SAMPLES_PER_CLASS} negative)")

    print("Applying text preprocessing (cleaning, tokenization, stopword removal)...")
    df_processed = apply_preprocessing(df_balanced, text_col='comment_text')

    print("Performing 80/20 stratified train/test split...")
    train_df, test_df = train_test_split(
        df_processed,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_processed['sentiment']
    )
    print(f"Split complete: {len(train_df)} train samples, {len(test_df)} test samples")

    print("Saving processed datasets to data/processed/...")

    balanced_path = os.path.join(PROCESSED_DIR, 'balanced_dataset.csv')
    df_processed.to_csv(balanced_path, index=False)

    train_path = os.path.join(PROCESSED_DIR, 'train.csv')
    test_path = os.path.join(PROCESSED_DIR, 'test.csv')
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Dataset preparation completed successfully!")
    print(f"   - Full dataset: {balanced_path}")
    print(f"   - Train split:  {train_path}")
    print(f"   - Test split:   {test_path}")

if __name__ == "__main__":
    main()