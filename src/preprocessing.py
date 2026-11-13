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


if __name__ == "__main__":
    sample_text = "This is an AMAZING day! I'm so happy :) #blessed"
    print(f"Original: {sample_text}")
    print(f"Cleaned: {clean_text(sample_text)}")
    print(f"Preprocessed: {preprocess_pipeline(sample_text)}")