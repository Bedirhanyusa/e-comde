import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TurkishReviewDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: AutoTokenizer,
        max_len: int = 128,
    ):
        self.labels = labels
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[dict, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return item, label
