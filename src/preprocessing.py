"""
Text preprocessing utilities for GoEmotions dataset.
Handles tokenization, cleaning, and data preparation for sentiment analysis.
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from typing import List

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """
    Clean and normalize text data.

    Steps:
    1. Convert to lowercase
    2. Remove punctuation and special characters
    3. Remove extra whitespace

    Args:
        text: Raw text string

    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r'[^\w\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize_and_remove_stopwords(text: str) -> List[str]:
    """
    Tokenize text and remove English stopwords.

    Args:
        text: Input text string (should be pre-cleaned)

    Returns:
        List of tokens without stopwords
    """
    tokens = text.split()
    return [token for token in tokens if token not in STOPWORDS]


def preprocess_pipeline(text: str) -> str:
    """
    Full preprocessing pipeline: clean, tokenize, remove stopwords, rejoin.

    Args:
        text: Raw text string

    Returns:
        Preprocessed text as a single string
    """
    cleaned = clean_text(text)
    tokens = tokenize_and_remove_stopwords(cleaned)
    return " ".join(tokens)


def apply_preprocessing(df: pd.DataFrame, text_col: str = 'comment_text') -> pd.DataFrame:
    """
    Apply preprocessing pipeline to a DataFrame column.

    Args:
        df: Input DataFrame with text column
        text_col: Name of the column containing text

    Returns:
        DataFrame with new columns: 'cleaned_text', 'tokens', 'text_length'
    """
    df_processed = df.copy()

    df_processed['cleaned_text'] = df_processed[text_col].apply(clean_text)

    df_processed['tokens'] = df_processed['cleaned_text'].apply(tokenize_and_remove_stopwords)

    df_processed['text_length'] = df_processed['tokens'].apply(len)

    return df_processed


def load_goemotions_tsv(file_path: str) -> pd.DataFrame:
    """
    Load a GoEmotions TSV file.

    Args:
        file_path: Path to the .tsv file

    Returns:
        DataFrame with columns: 'comment_text', 'emotion_ids', 'id'
    """
    df = pd.read_csv(file_path, sep='\t', header=None, names=['comment_text', 'emotion_ids', 'id'])

    df['emotion_ids'] = df['emotion_ids'].apply(
        lambda x: [int(e) for e in str(x).split(',')] if pd.notna(x) and str(x) != '' else []
    )

    return df

POSITIVE_EMOTIONS = {1, 2, 5, 7, 8, 11, 13, 15, 17, 19, 21, 25}
NEGATIVE_EMOTIONS = {0, 3, 4, 6, 9, 10, 12, 14, 16, 18, 20}
AMBIGUOUS_EMOTIONS = {22, 23, 24, 26}

def map_emotion_to_sentiment(emotion_id: int) -> str:
    """
    Map a single emotion ID to its sentiment group.

    Args:
        emotion_id: Integer ID of the emotion (0-27)

    Returns:
        Sentiment group: 'positive', 'negative', 'ambiguous', or 'neutral'
    """
    if emotion_id in POSITIVE_EMOTIONS:
        return 'positive'
    elif emotion_id in NEGATIVE_EMOTIONS:
        return 'negative'
    elif emotion_id in AMBIGUOUS_EMOTIONS:
        return 'ambiguous'
    elif emotion_id == 27:
        return 'neutral'
    else:
        return 'unknown'


def classify_comment_sentiment(emotion_ids: list) -> str:
    """
    Classify a comment's overall sentiment based on its emotion IDs.

    A comment is classified as 'positive' or 'negative' ONLY if ALL its emotions
    belong to the same group. Mixed sentiments are labeled as 'mixed'.

    Args:
        emotion_ids: List of emotion IDs for a single comment

    Returns:
        Sentiment label: 'positive', 'negative', 'ambiguous', 'neutral', 'mixed', or 'unknown'
    """
    if not emotion_ids:
        return 'neutral'

    sentiments = [map_emotion_to_sentiment(eid) for eid in emotion_ids]

    sentiments = [s for s in sentiments if s != 'unknown']

    if not sentiments:
        return 'unknown'

    unique_sentiments = set(sentiments)

    if len(unique_sentiments) == 1:
        return unique_sentiments.pop()
    else:
        return 'mixed'


def filter_goemotions_for_binary_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter GoEmotions dataset for binary sentiment classification.

    Steps:
    1. Map each comment to a sentiment group based on official mapping
    2. Keep ONLY comments where ALL emotions belong to the same group
    3. Remove neutral, ambiguous, and mixed-sentiment comments
    4. Remove duplicate comments

    Args:
        df: DataFrame with columns 'comment_text' and 'emotion_ids'

    Returns:
        Filtered DataFrame with only pure positive and negative comments
    """
    df_filtered = df.copy()

    df_filtered['sentiment'] = df_filtered['emotion_ids'].apply(classify_comment_sentiment)

    df_filtered = df_filtered[df_filtered['sentiment'].isin(['positive', 'negative'])].copy()

    df_filtered = df_filtered.drop_duplicates(subset=['comment_text'])

    df_filtered = df_filtered.reset_index(drop=True)

    return df_filtered


def create_balanced_binary_dataset(df: pd.DataFrame, n_samples_per_class: int = 500,
                                   random_state: int = 42) -> pd.DataFrame:
    """
    Create a balanced binary dataset with equal samples from each class.

    Args:
        df: Filtered DataFrame with 'sentiment' column (only 'positive' and 'negative')
        n_samples_per_class: Number of samples to keep from each class
        random_state: Random seed for reproducibility

    Returns:
        Balanced DataFrame with exactly n_samples_per_class from each sentiment
    """
    balanced_dfs = []

    for sentiment in ['positive', 'negative']:
        df_class = df[df['sentiment'] == sentiment]

        if len(df_class) < n_samples_per_class:
            raise ValueError(f"Not enough {sentiment} samples. Found {len(df_class)}, need {n_samples_per_class}")

        df_sampled = df_class.sample(n=n_samples_per_class, random_state=random_state)
        balanced_dfs.append(df_sampled)

    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df_balanced


def load_and_prepare_goemotions(raw_dir: str, n_samples_per_class: int = 500,
                                random_state: int = 42) -> pd.DataFrame:
    """
    Complete pipeline: load GoEmotions, filter, and balance for binary classification.

    Args:
        raw_dir: Path to directory containing train.tsv, dev.tsv, test.tsv
        n_samples_per_class: Number of samples per class in final balanced dataset
        random_state: Random seed for reproducibility

    Returns:
        Balanced DataFrame ready for train/test split
    """
    import os

    dfs = []
    for split in ['train', 'dev', 'test']:
        file_path = os.path.join(raw_dir, f'{split}.tsv')
        if os.path.exists(file_path):
            df_split = load_goemotions_tsv(file_path)
            dfs.append(df_split)

    if not dfs:
        raise FileNotFoundError(f"No .tsv files found in {raw_dir}")

    df_full = pd.concat(dfs, ignore_index=True)

    df_filtered = filter_goemotions_for_binary_classification(df_full)

    df_balanced = create_balanced_binary_dataset(
        df_filtered,
        n_samples_per_class=n_samples_per_class,
        random_state=random_state
    )

    return df_balanced


if __name__ == "__main__":
    sample_text = "This is an AMAZING day! I'm so happy :) #blessed"
    print(f"Original: {sample_text}")
    print(f"Cleaned: {clean_text(sample_text)}")
    print(f"Preprocessed: {preprocess_pipeline(sample_text)}")

    print("\n--- Sentiment Classification Tests ---")
    test_cases = [
        ([1, 2], "joy + excitement"),
        ([0, 3], "anger + fear"),
        ([1, 0], "joy + anger (MIXED)"),
        ([22, 23], "realization + surprise (AMBIGUOUS)"),
        ([27], "neutral"),
        ([], "empty (neutral)")
    ]

    for emotion_ids, description in test_cases:
        sentiment = classify_comment_sentiment(emotion_ids)
        print(f"{description:30s} → {sentiment}")
