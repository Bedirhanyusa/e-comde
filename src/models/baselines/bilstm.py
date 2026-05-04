import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from src.evaluation.metrics import compute_classification_metrics


class BiLSTMDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: dict, max_len: int = 128):
        self.labels = labels
        self.sequences = []
        for text in texts:
            tokens = text.lower().split()
            ids = [vocab.get(t, 1) for t in tokens[:max_len]]
            ids += [0] * (max_len - len(ids))
            self.sequences.append(ids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 100, hidden_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(emb)
        out = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return self.fc(self.dropout(out))


def build_vocab(texts: list[str], min_freq: int = 2) -> dict:
    from collections import Counter
    counts = Counter(token for text in texts for token in text.lower().split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, freq in counts.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def train_and_evaluate(
    train_csv: str = "data/processed/train.csv",
    test_csv: str = "data/processed/test.csv",
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    vocab = build_vocab(train_df["Review"].tolist())

    train_ds = BiLSTMDataset(train_df["Review"].tolist(), train_df["label"].tolist(), vocab)
    test_ds = BiLSTMDataset(test_df["Review"].tolist(), test_df["label"].tolist(), vocab)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = BiLSTMClassifier(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    all_preds = []
    with torch.no_grad():
        for x, _ in test_loader:
            all_preds.extend(model(x.to(device)).argmax(dim=-1).tolist())

    metrics = compute_classification_metrics(test_df["label"].tolist(), all_preds)
    print(f"BiLSTM | Dogruluk: {metrics['accuracy']:.4f} | Makro-F1: {metrics['macro_f1']:.4f}")
    return metrics


if __name__ == "__main__":
    train_and_evaluate()
