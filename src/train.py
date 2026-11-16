"""Training entry point for sentiment analysis models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Any, Optional
import copy
import time


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        lr: float = 0.001,
        patience: int = 3,
        device: str = "cpu",
) -> Dict[str, Any]:
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device).float()

            if labels.dim() == 1:
                labels = labels.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)

            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = epoch_train_loss / max(n_train_batches, 1)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input_ids"].to(device)
                labels = batch["labels"].to(device).float()

                if labels.dim() == 1:
                    labels = labels.unsqueeze(1)

                outputs = model(inputs)

                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(1)

                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = epoch_val_loss / max(n_val_batches, 1)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    training_time = time.time() - start_time

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "epochs_run": len(train_losses),
        "training_time": training_time,
        "best_model_state": best_model_state,
    }

    return history


if __name__ == "__main__":
    from torch.utils.data import Dataset

    class DummyDataset(Dataset):
        def __init__(self, n_samples: int, seq_len: int, vocab_size: int):
            self.data = torch.randint(0, vocab_size, (n_samples, seq_len))
            self.labels = torch.randint(0, 2, (n_samples,))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return {"input_ids": self.data[idx], "labels": self.labels[idx]}

    from models import LSTMClassifier

    train_dataset = DummyDataset(n_samples=100, seq_len=20, vocab_size=1000)
    val_dataset = DummyDataset(n_samples=30, seq_len=20, vocab_size=1000)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    model = LSTMClassifier(
        vocab_size=1000,
        embedding_dim=32,
        hidden_dim=64,
        output_dim=1,
        num_layers=1,
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        lr=0.01,
        patience=2,
    )

    print(f"Epochs run: {history['epochs_run']}")
    print(f"Best val loss: {history['best_val_loss']:.4f}")
    print(f"Training time: {history['training_time']:.2f}s")
    print(f"Train losses: {[f'{x:.4f}' for x in history['train_losses']]}")
    print(f"Val losses: {[f'{x:.4f}' for x in history['val_losses']]}")