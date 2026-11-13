# GoEmotions Dataset

## Source
- **Name**: GoEmotions
- **Authors**: Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020)
- **Publication**: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020)
- **Link**: [Kaggle Dataset](https://www.kaggle.com/datasets/debarshichanda/goemotions)
- **Paper**: [arXiv:2005.00547](https://arxiv.org/abs/2005.00547)

## Description
GoEmotions is a corpus of 58,009 English Reddit comments labeled by human raters with 27 emotion categories or Neutral. Comments are short (at most 30 tokens, ~13 words on average) and can have multiple emotion labels. Person names and religion terms are masked with `[NAME]` and `[RELIGION]`.

For this project, we use the **filtered version** provided by the authors, which includes only examples where at least 2 raters agreed on the label. This version contains 54,263 comments split into:
- `train.tsv`
- `dev.tsv`
- `test.tsv`

Each row contains: `comment_text`, `emotion_ids` (tab-separated), and `id`.

## Official Sentiment Mapping
The 27 emotions are mapped into three main sentiment groups (plus Neutral):
- **Positive** (12): amusement, excitement, joy, love, desire, optimism, caring, pride, admiration, gratitude, relief, approval.
- **Negative** (11): fear, nervousness, remorse, embarrassment, disappointment, sadness, grief, disgust, anger, annoyance, disapproval.
- **Ambiguous** (4): realization, surprise, curiosity, confusion.
- **Neutral**: No emotion expressed.

## Data Preparation for this Project
1. Download the dataset from Kaggle and place the `.tsv` files in the `data/raw/` directory.
2. Apply the official sentiment mapping.
3. Keep **only** comments where ALL emotions belong to the same group (pure positive or pure negative).
4. Filter out neutral, ambiguous, and mixed-sentiment comments.
5. Remove duplicated comments.
6. Construct a **balanced dataset** of 500 positive and 500 negative comments (randomly sampled with `random_state=42`).

## Directory Structure
- `raw/`: Contains the original `.tsv` files (ignored by Git due to size limits).
- `processed/`: Will contain the final balanced and preprocessed dataset (e.g., `balanced_dataset.csv`, `train.csv`, `test.csv`).