# GoEmotions Dataset — Project Data Documentation

## 1. Source and Citation
- **Name**: GoEmotions
- **Authors**: Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020)
- **Publication**: *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020)*, pp. 4040–4054
- **Links**: [ACL Anthology](https://aclanthology.org/2020.acl-main.372/) | [arXiv:2005.00547](https://arxiv.org/abs/2005.00547) | [Kaggle Dataset](https://www.kaggle.com/datasets/google-research/goemotions)

## 2. Dataset Overview
GoEmotions is a corpus of 58,009 English Reddit comments labeled by human raters across **27 emotion categories** plus **Neutral**. Comments are short (at most ~30 tokens, ~12–13 words on average) and may carry multiple emotion annotations. Proper nouns and religious terms are masked with `[NAME]` and `[RELIGION]`.

For this project, we use the official **filtered splits** (`train.tsv`, `dev.tsv`, `test.tsv`), where at least two raters agreed on the annotation. Each row is tab-separated with three fields: `comment_text`, `emotion_ids` (comma-separated integer IDs `0–27`), and `id`.

## 3. Official Emotion-to-Sentiment Mapping (`src/preprocessing.py`)
The 27 emotion IDs are grouped into three sentiment categories plus Neutral:
- **Positive (12 emotions — IDs `{1, 2, 5, 7, 8, 11, 13, 15, 17, 19, 21, 25}`)**: amusement, excitement, joy, love, desire, optimism, caring, pride, admiration, gratitude, relief, approval.
- **Negative (11 emotions — IDs `{0, 3, 4, 6, 9, 10, 12, 14, 16, 18, 20}`)**: anger, annoyance, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness.
- **Ambiguous (4 emotions — IDs `{22, 23, 24, 26}`)**: realization, surprise, curiosity, confusion.
- **Neutral (ID `27`)**: neutral.

## 4. Preprocessing & Balancing Pipeline (`src/build_dataset.py`)
1. **Pure-Sentiment Filtering**: Load `train.tsv`, `dev.tsv`, and `test.tsv` from `data/raw/`. Keep **only** comments where **all** annotated emotion IDs belong exclusively to either `positive` or `negative`. Discard neutral, ambiguous, mixed-sentiment, and duplicate comments.
   - **Filtered result**: **29,991 valid binary comments** (**20,777 positive** and **9,214 negative**), saved locally to `data/processed/filtered_dataset.csv`.
2. **Class Balancing**: Randomly sample **500 positive** and **500 negative** comments using `random_state = 42` → **1,000 comments total**, saved to `data/processed/balanced_dataset.csv`.
3. **Text Cleaning & Tokenization**: Lowercase text, remove punctuation/special characters via regex, and strip English stopwords (`cleaned_text`, `tokens`, `text_length`).
4. **Stratified Train/Test Split**: Partition the 1,000-comment balanced dataset using an **80/20 stratified split** (`random_state = 42`):
   - `data/processed/train.csv` — **800 comments** (400 positive, 400 negative)
   - `data/processed/test.csv` — **200 comments** (100 positive, 100 negative)

## 5. Directory Layout
```text
data/
├── raw/                         # Official GoEmotions TSV splits (not versioned in Git)
│   ├── train.tsv
│   ├── dev.tsv
│   └── test.tsv
├── processed/                   # Generated balanced & split CSV files (regenerated via build_dataset.py)
│   ├── filtered_dataset.csv
│   ├── balanced_dataset.csv
│   ├── train.csv
│   └── test.csv
└── README.md
```
