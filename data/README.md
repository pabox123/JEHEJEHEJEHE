# Dataset

This project uses **GoEmotions**, a collection of English Reddit comments annotated with fine-grained emotions. The dataset was introduced by Demszky et al. (2020) and is available through [Kaggle](https://www.kaggle.com/datasets/google-research/goemotions).

## Dataset overview

GoEmotions contains short comments with 27 emotion labels and a neutral label. For this project, the labels are grouped into positive and negative sentiment according to the implemented emotion mapping. Neutral, ambiguous, and mixed-emotion comments are excluded from the binary classification task.

The final balanced project dataset is:

- 500 positive comments
- 500 negative comments
- 1,000 comments in total
- `text`: the original Reddit comment
- `label`: `0` for negative and `1` for positive

## Preprocessing workflow

The implemented workflow is:

1. Download the filtered GoEmotions files (`train.tsv`, `dev.tsv`, and `test.tsv`).
2. Apply the official emotion-to-sentiment mapping.
3. Keep comments whose emotion labels belong consistently to the positive or negative group.
4. Remove neutral, ambiguous, mixed-emotion, and duplicate comments.
5. Sample 500 comments from each sentiment class using random seed `42`.
6. Save the result as `data/processed/balanced_dataset.csv`.

The raw files and the processed CSV are intentionally excluded from version control. Each person reproducing the project should generate the dataset locally and confirm its provenance before training a model.

After loading the three official TSV files, the filtering step produced **29,991 valid binary comments**:

- 9,214 negative comments
- 20,777 positive comments

The balancing step then randomly sampled 500 comments from each class with seed 42 and saved 1,000 comments in `data/processed/balanced_dataset.csv`.

## Directory layout

```text
data/
|-- raw/                         # Downloaded GoEmotions files (not versioned)
|   |-- train.tsv
|   |-- dev.tsv
|   `-- test.tsv
|-- processed/                   # Generated project dataset (not versioned)
|   `-- balanced_dataset.csv
`-- README.md
```

The notebooks and source files document the preprocessing and modeling workflow. Notebook 04 creates a synthetic development dataset only when the raw GoEmotions TSV files and the balanced CSV are unavailable.
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
