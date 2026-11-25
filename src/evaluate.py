"""Evaluation utilities for trained sentiment analysis models."""

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict, Any


def evaluate_model(
        model: nn.Module,
        data_loader: DataLoader,
        device: str = "cpu",
        threshold: float = 0.5,
) -> Dict[str, Any]:
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(inputs)

            if outputs.dim() > 1 and outputs.size(-1) == 1:
                outputs = outputs.squeeze(-1)

            probs = torch.sigmoid(outputs)
            preds = (probs >= threshold).long()

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
        "kappa": float(cohen_kappa_score(all_labels, all_preds)),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        "predictions": all_preds.tolist(),
        "labels": all_labels.tolist(),
    }

    return metrics


def print_metrics(metrics: Dict[str, Any], model_name: str = "Model") -> None:
    print(f"\n{'=' * 50}")
    print(f" Evaluation Metrics - {model_name}")
    print(f"{'=' * 50}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-score : {metrics['f1']:.4f}")
    print(f"  Kappa    : {metrics['kappa']:.4f}")
    print(f"  Confusion Matrix:")
    for row in metrics["confusion_matrix"]:
        print(f"    {row}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    from torch.utils.data import Dataset
    from models import LSTMClassifier

    class DummyDataset(Dataset):
        def __init__(self, n_samples: int, seq_len: int, vocab_size: int):
            self.data = torch.randint(0, vocab_size, (n_samples, seq_len))
            self.labels = torch.randint(0, 2, (n_samples,))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return {"input_ids": self.data[idx], "labels": self.labels[idx]}

    test_dataset = DummyDataset(n_samples=50, seq_len=20, vocab_size=1000)
    test_loader = DataLoader(test_dataset, batch_size=16)

    model = LSTMClassifier(
        vocab_size=1000,
        embedding_dim=32,
        hidden_dim=64,
        output_dim=1,
        num_layers=1,
    )

    metrics = evaluate_model(model=model, data_loader=test_loader)
    print_metrics(metrics, model_name="LSTMClassifier (untrained)")



def evaluate_transformer(model, test_loader, device='cpu'):
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='binary'),
        'recall': recall_score(all_labels, all_preds, average='binary'),
        'f1': f1_score(all_labels, all_preds, average='binary'),
        'kappa': cohen_kappa_score(all_labels, all_preds),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'classification_report': classification_report(
            all_labels, all_preds, target_names=['negative', 'positive']
        )
    }

    return metrics, all_preds, all_labels