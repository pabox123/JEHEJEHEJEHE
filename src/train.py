"""Training entry point for sentiment analysis models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any
import copy
import time
from tqdm import tqdm


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


def train_transformer(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 5,
        lr: float = 2e-5,
        device: str = "cpu",
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | '
              f'Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            print(f'  -> new best model saved (Val Acc: {best_val_acc:.4f})')

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


if __name__ == "__main__":
    from torch.utils.data import Dataset
    from src.models import LSTMClassifier

    class DummyDataset(Dataset):
        def __init__(self, n_samples: int, seq_len: int, vocab_size: int):
            self.data = torch.randint(0, vocab_size, (n_samples, seq_len))
            self.labels = torch.randint(0, 2, (n_samples,))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return {"input_ids": self.data[idx], "labels": self.labels[idx]}

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