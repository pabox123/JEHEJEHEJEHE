# CED-III-IT2-JEHEJEHEJEHE

## Sentiment Analysis on GoEmotions Dataset

Academic machine learning project for binary sentiment classification of Reddit comments using a balanced subset of the GoEmotions dataset, connecting sequence processing and neural memory architectures to Turing Machine computability concepts (OE3.3–OE3.6).

## Team Members

- Valeria Piza Saavedra - A00405037
- Krystal Angely Giraldo Ramirez - A00407967
- Miguel Jose Pabon Benavidez - A00406585

## Project Status: Complete

All data preparation stages, exploratory analyses, baseline evaluations, classical neural architectures, hyperparameter tuning, and the Transformer extension have been implemented, executed, and documented:

1. **Dataset & Preprocessing (`src/preprocessing.py`, `src/build_dataset.py`, `notebooks/01_eda.ipynb`)**:
   - Filtered the three official GoEmotions splits (`train.tsv`, `dev.tsv`, `test.tsv`) to retain only pure positive or pure negative comments, excluding neutral, ambiguous, and mixed-emotion entries (**29,991 consistent comments**: 20,777 positive and 9,214 negative).
   - Sampled a reproducible balanced subset of **1,000 comments** (500 positive, 500 negative) with `random_state=42`, partitioned into an **80/20 stratified train/test split** (800 train / 200 test).
   - Applied lowercasing, punctuation removal, and English/Reddit stopword filtering (`cleaned_text`).

2. **Model Implementations (`src/models.py`, `src/train.py`, `src/evaluate.py`, `src/visualize.py`)**:
   - **Dummy Baselines (`notebooks/02_baseline_model.ipynb`)**: `most_frequent` and `stratified` strategies establishing the 50% accuracy and $\kappa = 0.000$ statistical floor.
   - **Classical Neural Models (`notebooks/03_dense_rnn_lstm.ipynb`)**:
     - `DenseNN`: 3-layer MLP (`2253 → 256 → 128 → 1`) on Bag-of-Words features (`CountVectorizer`, 2,253 terms).
     - `VanillaRNN`: 1-layer bidirectional RNN (`embed=64`, `hidden=128` per direction) on 30-token padded sequences.
     - `LSTMClassifier`: 2-layer stacked bidirectional LSTM (`embed=64`), tuned via a 12-combination hyperparameter grid search (`hidden_dim ∈ {64, 128, 256}`, `lr ∈ {1e-3, 5e-4}`, `batch_size ∈ {32, 64}`).
   - **Transformer Extension (`notebooks/04_transformer_extension.ipynb`)**:
     - `TransformerSentiment`: Fine-tuned `distilbert-base-uncased` using `DistilBertTokenizerFast` (`max_length=64`), `[CLS]` representation, custom classification head, `AdamW` (`lr=2e-5`), and gradient clipping (`max_norm=1.0`).

## Summary of Experimental Results (200-Comment Test Split)

| Model Architecture | Accuracy | Precision | Recall | F1-Score | Cohen's Kappa ($\kappa$) | Confusion Matrix |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Most Frequent)** | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | `[[100, 0], [100, 0]]` |
| **Dummy (Stratified)** | 0.500 | 0.500 | 0.510 | 0.505 | 0.000 | `[[49, 51], [49, 51]]` |
| **Dense NN (BoW)** | **0.745** | **0.738** | **0.760** | **0.749** | **0.490** | `[[73, 27], [24, 76]]` |
| **Vanilla RNN (Bi)** | 0.550 | 0.536 | 0.740 | 0.622 | 0.100 | `[[36, 64], [26, 74]]` |
| **LSTM (Initial, 15 ep)** | 0.555 | 0.580 | 0.400 | 0.473 | 0.110 | `[[71, 29], [60, 40]]` |
| **LSTM (Tuned Final)** | 0.525 | 0.520 | 0.660 | 0.582 | 0.050 | `[[39, 61], [34, 66]]` |
| **DistilBERT (Extension)** | **0.775** | **0.762** | **0.800** | **0.780** | **0.550** | `[[75, 25], [20, 80]]` |

## Repository Structure

- `data/` — Dataset documentation (`data/README.md`); raw `.tsv` and processed `.csv` splits are generated locally via `python src/build_dataset.py`.
- `notebooks/` — Ordered Jupyter notebooks (`01_eda.ipynb` through `04_transformer_extension.ipynb`).
- `src/` — Modular Python package (`preprocessing.py`, `build_dataset.py`, `models.py`, `train.py`, `evaluate.py`, `visualize.py`).
- `outputs/figures/` — All 17 generated EDA, training curve, confusion matrix, and model comparison plots.
- `outputs/metrics/` — Tracked CSV and JSON evaluation metrics across all models.
- `docs/` — Final academic report (`report.md`, `report.pdf`), slide deck (`presentation.md`, `presentation.pdf`), and bibliography (`references.bib`).
- `logs/` — Detailed experimental log (`experiment_notes.md`) and training execution summary (`training_logs.txt`).
- `prompts/` — Individual Level-3 GenAI interaction tables and verbatim prompt logs per team member.
