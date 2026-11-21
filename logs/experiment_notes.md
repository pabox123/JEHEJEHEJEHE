# Experiment Notes

This log records the experiments that have actually been run, along with decisions that affect how their results should be interpreted.

  ---

## Dummy Baseline (DummyClassifier) — November 16, 2026

Implemented in `notebooks/02_baseline_model.ipynb` (commit 7be0b1d).

### Configuration
- **Model:** `sklearn.dummy.DummyClassifier(strategy="most_frequent")`
- **Data:** Balanced GoEmotions subset (1,000 comments: 500 negative, 500 positive)
- **Split:** 80/20 stratified (800 train / 200 test), seed 42
- **Input:** BoW vectors (CountVectorizer, max_features=5000, stopwords EN) — 2,253 dimensions

### Recorded Result
`outputs/metrics/baseline_metrics.csv` reports:
- Accuracy: 0.500
- Precision: 0.333
- Recall: 0.500
- F1-score: 0.400
- Cohen's Kappa: 0.000

### Interpretation
The majority-class baseline (always predicts "positive") achieves 50% accuracy but zero kappa, confirming the balanced split. This establishes the floor for all subsequent
models.

  ---

## Dense Neural Network (BoW) — November 18, 2026

Implemented in `notebooks/03_dense_rnn_lstm.ipynb` (validated in commit d25bdf0, trained Nov 18).

### Architecture (`src/models.py:DenseNN`, commit b6931d6 Nov 13)
| Layer | Configuration |
  |-------|---------------|
| Input | BoW vector, 2,253 dimensions |
| Hidden 1 | Linear(2253 → 256) + ReLU + Dropout(0.3) |
| Hidden 2 | Linear(256 → 128) + ReLU + Dropout(0.3) |
| Output | Linear(128 → 1) + sigmoid |

### Training (`src/train.py:train_model`, commit 4bb6ce5 Nov 16)
- **Epochs:** 20 (early stopping patience 4)
- **Learning rate:** 1e-3
- **Batch size:** 64
- **Optimizer:** Adam
- **Loss:** BCEWithLogitsLoss
- **Device:** CPU

### Recorded Result
`outputs/metrics/model_comparison.csv` (DenseNN row):
- Accuracy: 0.745
- Precision: 0.738
- Recall: 0.760
- F1-score: 0.749
- Cohen's Kappa: 0.490

Confusion matrix: `[[73, 27], [24, 76]]`

Figures saved:
- `outputs/figures/dense_training_curves.png`
- `outputs/figures/dense_confusion_matrix.png`

  ---

## Vanilla RNN — November 18, 2026

Added to `notebooks/03_dense_rnn_lstm.ipynb` (commit 069bdd2).

### Architecture (`src/models.py:VanillaRNN`, commit f4e8c1e Nov 14)
| Layer | Configuration |
  |-------|---------------|
| Embedding | vocab 2,254 → 64 |
| Recurrent | 1 bidirectional vanilla RNN, hidden 128 per direction |
| Pooling | Concatenation of forward/backward final states (256) |
| Head | Dropout(0.3) + Linear(256 → 1) + sigmoid |

### Training
- **Epochs:** 15 (early stopping patience 3)
- **Learning rate:** 1e-3
- **Batch size:** 64
- **Sequence length:** 30 tokens (left-padded, PAD_IDX=0)

### Recorded Result
`outputs/metrics/model_comparison.csv` (VanillaRNN row):
- Accuracy: 0.550
- Precision: 0.536
- Recall: 0.740
- F1-score: 0.622
- Cohen's Kappa: 0.100

Confusion matrix: `[[36, 64], [26, 74]]`

Figures saved:
- `outputs/figures/rnn_training_curves.png`
- `outputs/figures/rnn_confusion_matrix.png`

### Observation
Strong positive-class bias (recall 0.74, precision 0.54, kappa 0.10). Most negative comments misclassified as positive — typical of underfitting on small data with vanishing
gradients.

  ---

## LSTM — November 18–19, 2026

Added to `notebooks/03_dense_rnn_lstm.ipynb` (commit 069bdd2 Nov 18, hyperparameter tuning commit ae764c4 Nov 19).

### Architecture (`src/models.py:LSTMClassifier`, commit b660fdd Nov 15)
| Layer | Configuration |
  |-------|---------------|
| Embedding | vocab 2,254 → 64 |
| Recurrent | 2 stacked bidirectional LSTM layers, hidden 128 per direction |
| Pooling | Concatenation of forward/backward final states |
| Head | Dropout(0.3) + Linear → 1 + sigmoid |

### Initial Training (Nov 18)
- **Epochs:** 15 (early stopping patience 3)
- **Learning rate:** 1e-3
- **Batch size:** 64
- **Sequence length:** 30

### Hyperparameter Tuning — November 19, 2026 (commit ae764c4)
Manual grid search over 12 combinations (3 hidden_dims × 2 learning_rates × 2 batch_sizes), 10 epochs each, patience 3, ranked by F1-score.

| hidden_dim | lr | batch_size | Accuracy | F1 | Kappa |
  |------------|-----|------------|----------|-----|-------|
| 64 | 0.001 | 32 | 0.680 | **0.692** | 0.36 |
| 128 | 0.0005 | 32 | 0.665 | 0.663 | 0.33 |
| 256 | 0.0005 | 64 | 0.495 | 0.660 | -0.01 |

Best: **hidden_dim=64, lr=0.001, batch_size=32** (F1=0.6923)

Logged to: `logs/hyperparameter_search.csv` (via `src/train.py:log_hyperparameter_run`)

### Final Retraining (Best Config, 20 epochs, patience 4)
`outputs/metrics/lstm_tuned_metrics.json`:
- Accuracy: 0.525
- Precision: 0.520
- Recall: 0.660
- F1-score: 0.582
- Cohen's Kappa: 0.050

Confusion matrix: `[[39, 61], [34, 66]]`

Figures saved:
- `outputs/figures/lstm_training_curves.png`
- `outputs/figures/lstm_confusion_matrix.png`

### Critical Note
Despite the best search score (F1=0.692), the final retraining collapsed to near-chance kappa (0.05). Diagnosed as: (a) early stopping restoring a weak checkpoint on this small
dataset, (b) 800 training comments insufficient for 2-layer bidirectional LSTM to overcome frequency bias. Documented honestly in comparative table — LSTM underperforms DenseNN
on this data.

  ---

## Model Comparison Summary — November 20, 2026

Consolidated in `outputs/metrics/model_comparison.csv` / `.json` (commit 38f8699).

| Model | Accuracy | Precision | Recall | F1-Score | Cohen's Kappa |
  |-------|----------|-----------|--------|----------|---------------|
| Dense NN (BoW) | 0.745 | 0.738 | 0.760 | 0.749 | 0.490 |
| Vanilla RNN | 0.550 | 0.536 | 0.740 | 0.622 | 0.100 |
| LSTM (tuned) | 0.525 | 0.520 | 0.660 | 0.582 | 0.050 |

Comparison figures:
- `outputs/figures/comp_accuracy.png`
- `outputs/figures/comp_f1.png`
- `outputs/figures/comp_kappa.png`
- `outputs/figures/model_comparison_accuracy.png`

Key finding: **Dense BoW network is the strongest classical model** — outperforms both sequential models on every metric. Consistent with short GoEmotions comments (≈13 words):
sentiment carried by keywords, BoW avoids sequential optimization difficulty.

  ---

## DistilBERT (Transformer Extension) — November 17, 2026

Implemented in `notebooks/04_transformer_extension.ipynb` (merge commit ca58fc5 Nov 16, run Nov 17).

### Configuration
- **Backbone:** `distilbert-base-uncased`
- **Tokenizer:** `DistilBertTokenizerFast`
- **Epochs:** 5
- **Learning rate:** 2e-5
- **Batch size:** 16
- **Optimizer:** AdamW
- **Loss:** Cross-entropy
- **Gradient clipping:** max norm 1.0
- **Max sequence length:** 64 tokens
- **Split:** 80/20 stratified (800/200), seed 42
- **Hardware:** CPU on Windows

### Recorded Result
`outputs/metrics/transformer_metrics.json`:
- Accuracy: 0.775
- Precision: 0.762
- Recall: 0.800
- F1-score: 0.780
- Cohen's kappa: 0.550

Confusion matrix: `[[75, 25], [20, 80]]` (155/200 correct)

Figures saved:
- `outputs/figures/04_sequence_length_distribution.png`
- `outputs/figures/04_transformer_training_curves.png`
- `outputs/figures/04_transformer_confusion_matrix.png`

### Interpretation
Best overall model (Accuracy 0.775, Kappa 0.550). Result specific to the balanced 1,000-comment subset — **should not be generalized to full GoEmotions corpus**.

  ---

## Reproducibility Notes

- Random seed `42` used for balanced dataset creation and train/test split.
- Raw data, processed data, and large model weights are **not versioned**.
- Metrics stored as JSON/CSV in `outputs/metrics/`.
- Generated figures stored as PNG in `outputs/figures/`.
- Hyperparameter search trail in `logs/hyperparameter_search.csv`.
- AIGen interaction logs in `prompts/AIGEN_KRYSTAL/`.

  ---

## Implementation Timeline Summary

| Date | Milestone |
  |------|-----------|
| 2026-11-13 | DenseNN architecture (`src/models.py`) |
| 2026-11-14 | VanillaRNN architecture |
| 2026-11-15 | LSTMClassifier architecture |
| 2026-11-16 | Training loop with early stopping (`src/train.py`); Evaluation module (`src/evaluate.py`); Visualization utilities (`src/visualize.py`); DummyClassifier baseline |
| 2026-11-17 | Notebook 03 created; Data pipeline merged |
| 2026-11-18 | Dense, Vanilla RNN, LSTM experiments executed in notebook 03 |
| 2026-11-19 | LSTM hyperparameter grid search (12 combos); Final retraining; Metrics & figures saved |
| 2026-11-20 | AIGen interactions documented; Neural Models Architecture sections written in report |
| 2026-11-21 | Final corrections; All feature branches merged to develop |

  ---