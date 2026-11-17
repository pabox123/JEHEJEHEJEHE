# Metrics

This folder stores the evaluation results produced by the notebooks. Each JSON file should be read together with the experiment log because the dataset and split used for a run determine how its numbers should be interpreted.

## Current artifact

`transformer_metrics.json` contains the results of the DistilBERT run on the balanced 1,000-comment GoEmotions dataset:

```json
{
  "model": "DistilBERT",
  "accuracy": 0.775,
  "precision": 0.7619047619047619,
  "recall": 0.8,
  "f1": 0.7804878048780488,
  "kappa": 0.55,
  "epochs": 5,
  "learning_rate": 2e-05,
  "batch_size": 16,
  "max_length": 64
}
```

In this run, the test split contains 200 examples. The confusion matrix is `[[75, 25], [20, 80]]`: 155 comments were classified correctly and 45 were misclassified.

Future experiments should keep the same field names where possible and document the dataset source, split, random seed, and model configuration in `logs/experiment_notes.md`.
