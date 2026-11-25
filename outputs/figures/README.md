# Generated Figures

This directory stores all plots produced by the exploratory data analysis and model evaluation notebooks (`01_eda.ipynb` through `04_transformer_extension.ipynb`). All figures correspond to the balanced 1,000-comment GoEmotions dataset (`random_state = 42`, 800 train / 200 test split).

## 1. Exploratory Data Analysis (`notebooks/01_eda.ipynb`)
- `class_distribution.png` — Bar and pie charts confirming the 500 positive / 500 negative balanced class distribution.
- `text_length_distribution.png` — Word-length histograms and boxplots by sentiment class (mean ~12.2 words, max 29 words).
- `wordclouds.png` — Comparative word clouds for positive (green) and negative (red) Reddit comments after custom stopword removal.
- `top_words.png` — Top-20 most frequent words per sentiment class.

## 2. Dummy Baseline (`notebooks/02_baseline_model.ipynb`)
- `confusion_matrix_baseline.png` — Side-by-side test-set confusion matrices for `DummyClassifier(strategy="most_frequent")` (`[[100, 0], [100, 0]]`) and `DummyClassifier(strategy="stratified")` (`[[49, 51], [49, 51]]`).

## 3. Classical Neural Models (`notebooks/03_dense_rnn_lstm.ipynb`)
- `dense_training_curves.png` — Training and validation loss curves across 8 epochs for `DenseNN` (BoW).
- `dense_confusion_matrix.png` — Test-set confusion matrix for `DenseNN` (`[[73, 27], [24, 76]]`, Accuracy `0.745`, $\kappa = 0.490$).
- `rnn_training_curves.png` — Training and validation loss curves across 12 epochs for bidirectional `VanillaRNN`.
- `rnn_confusion_matrix.png` — Test-set confusion matrix for `VanillaRNN` (`[[36, 64], [26, 74]]`, Accuracy `0.550`, $\kappa = 0.100$).
- `lstm_training_curves.png` — Training and validation loss curves for bidirectional `LSTMClassifier`.
- `lstm_confusion_matrix.png` — Test-set confusion matrix for the tuned `LSTMClassifier` (`[[39, 61], [34, 66]]`, Accuracy `0.525`, $\kappa = 0.050$).

## 4. Transformer Extension (`notebooks/04_transformer_extension.ipynb`)
- `04_sequence_length_distribution.png` — Tokenized WordPiece sequence length distribution (`P95 = 33 tokens`, covered by `max_length = 64`).
- `04_transformer_training_curves.png` — Training and validation loss and accuracy curves across 5 fine-tuning epochs for `DistilBERT`.
- `04_transformer_confusion_matrix.png` — Test-set confusion matrix for `DistilBERT` (`[[75, 25], [20, 80]]`, Accuracy `0.775`, $\kappa = 0.550$).

## 5. Cross-Model Comparative Plots
- `comp_accuracy.png` — Comparative bar chart of Test Accuracy across all evaluated architectures (`Dummy` through `DistilBERT`).
- `comp_f1.png` — Comparative bar chart of Test F1-Score across all evaluated architectures.
- `comp_kappa.png` — Comparative bar chart of Test Cohen's Kappa ($\kappa$) across all evaluated architectures.
- `model_comparison_accuracy.png` — Summary accuracy comparison chart across classical neural architectures.
