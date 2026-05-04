import random
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from pathlib import Path
import pandas as pd

from src.data.dataset import TurkishReviewDataset
from src.training.trainer_config import TrainerConfig


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(path: str, text_col: str = "Review", label_col: str = "label"):
    df = pd.read_csv(path)
    return df[text_col].tolist(), df[label_col].tolist()


def train(cfg: TrainerConfig):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = BertForSequenceClassification.from_pretrained(cfg.model_name, num_labels=cfg.num_labels)
    model.to(device)

    train_texts, train_labels = load_split("data/processed/train.csv")
    val_texts, val_labels = load_split("data/processed/val.csv")

    train_ds = TurkishReviewDataset(train_texts, train_labels, tokenizer, cfg.max_len)
    val_ds = TurkishReviewDataset(val_texts, val_labels, tokenizer, cfg.max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    total_steps = (len(train_loader) // cfg.gradient_accumulation_steps) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_acc = 0.0
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, (batch, labels) in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = labels.to(device)

            outputs = model(**batch, labels=labels)
            loss = outputs.loss / cfg.gradient_accumulation_steps
            loss.backward()
            total_loss += outputs.loss.item()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch, labels in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = labels.to(device)
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()

        val_acc = correct / len(val_ds)
        print(f"Epoch {epoch + 1}/{cfg.num_epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(cfg.checkpoint_dir)
            tokenizer.save_pretrained(cfg.checkpoint_dir)
            print(f"  -> Checkpoint kaydedildi ({val_acc:.4f})")

    print(f"\nEgitim tamamlandi. En iyi val acc: {best_val_acc:.4f}")
    return best_val_acc


if __name__ == "__main__":
    cfg = TrainerConfig()
    train(cfg)
