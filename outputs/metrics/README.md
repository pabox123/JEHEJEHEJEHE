# Evaluation Metrics Artifacts

This directory contains the tracked quantitative evaluation results produced by `notebooks/02_baseline_model.ipynb`, `notebooks/03_dense_rnn_lstm.ipynb`, and `notebooks/04_transformer_extension.ipynb` on the held-out 200-comment stratified test split (`data/processed/test.csv`, `random_state = 42`).

## Tracked Files

1. **`baseline_metrics.csv`** — Test metrics for `DummyClassifier` baselines (`Most Frequent` and `Stratified`):
   - Both strategies achieve `Accuracy = 0.500` and `Cohens_Kappa = 0.000`.
2. **`model_comparison.csv` & `model_comparison.json`** — Consolidated test metrics for the classical neural models:
   - `DenseNN`: Accuracy `0.745`, Precision `0.7379`, Recall `0.7600`, F1 `0.7488`, Kappa `0.4900`
   - `VanillaRNN`: Accuracy `0.550`, Precision `0.5362`, Recall `0.7400`, F1 `0.6218`, Kappa `0.1000`
   - `LSTM_Final` (Tuned): Accuracy `0.525`, Precision `0.5197`, Recall `0.6600`, F1 `0.5815`, Kappa `0.0500`
3. **`lstm_tuned_metrics.json`** — Full evaluation artifact for the final retrained LSTM (`hidden_dim = 64, lr = 1e-3, batch_size = 32`), including confusion matrix `[[39, 61], [34, 66]]`, raw binary predictions, and ground-truth test labels.
4. **`transformer_metrics.json`** — Full evaluation artifact for the fine-tuned `DistilBERT` extension (`distilbert-base-uncased`, 5 epochs, `lr = 2e-5`, `batch_size = 16`, `max_length = 64`):
   - Accuracy `0.775`, Precision `0.7619`, Recall `0.8000`, F1 `0.7805`, Kappa `0.5500`, confusion matrix `[[75, 25], [20, 80]]`, and full per-class classification report.
