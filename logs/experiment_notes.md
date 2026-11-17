# Experiment Notes

This log records the experiments that have actually been run, along with decisions that affect how their results should be interpreted.

---

## DistilBERT run - November 17, 2026

This run used `data/processed/balanced_dataset.csv`, containing 1,000 comments sampled from the filtered GoEmotions dataset. The balanced file contains 500 negative and 500 positive examples.

### Configuration

- **Backbone:** `distilbert-base-uncased`
- **Epochs:** 5
- **Learning rate:** `2e-5`
- **Batch size:** 16
- **Optimizer:** AdamW
- **Loss:** cross-entropy
- **Gradient clipping:** maximum norm `1.0`
- **Split:** 80/20 stratified (800 training, 200 test examples)
- **Maximum sequence length:** 64 tokens
- **Hardware:** CPU on Windows

### Recorded result

`outputs/metrics/transformer_metrics.json` reports:

- Accuracy: 0.775
- Precision: 0.762
- Recall: 0.800
- F1-score: 0.780
- Cohen's kappa: 0.550

The confusion matrix is `[[75, 25], [20, 80]]`. The model classified 155 of 200 test comments correctly.

The run also produced the following figures:

- `outputs/figures/04_sequence_length_distribution.png`
- `outputs/figures/04_transformer_training_curves.png`
- `outputs/figures/04_transformer_confusion_matrix.png`

### Interpretation and next step

The result is specific to the balanced 1,000-comment subset and the recorded 80/20 split. It should not be presented as an evaluation of the complete GoEmotions dataset.

---

## Other models

The dummy baseline and the dense, RNN, and LSTM experiments are not implemented in the current repository. No metrics or model ranking are recorded for them.

---

## Reproducibility notes

- Use random seed `42` when creating the balanced dataset and splitting the data.
- Raw data, processed data, and large model weights are not versioned.
- Metrics are stored as JSON in `outputs/metrics/`.
- Generated figures are stored as PNG files in `outputs/figures/`.
