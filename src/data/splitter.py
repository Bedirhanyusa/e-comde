import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


def stratified_split(
    df: pd.DataFrame,
    text_col: str = "Review",
    label_col: str = "label",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
    output_dir: str = "data/processed",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_ratio), stratify=df[label_col], random_state=seed
    )
    relative_val = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - relative_val), stratify=temp_df[label_col], random_state=seed
    )

    for split, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        path = Path(output_dir) / f"{name}.csv"
        split[[text_col, label_col]].to_csv(path, index=False)
        counts = split[label_col].value_counts().sort_index().to_dict()
        print(f"{name:5s}: {len(split):5d} satir | {counts}")

    return train_df, val_df, test_df
