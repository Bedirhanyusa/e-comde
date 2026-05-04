import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

from src.evaluation.metrics import compute_classification_metrics


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50_000,
            analyzer="char_wb",
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=42,
        )),
    ])


def train_and_evaluate(
    train_csv: str = "data/processed/train.csv",
    test_csv: str = "data/processed/test.csv",
    save_path: str = "models/tfidf_lr.pkl",
):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    pipeline = build_pipeline()
    pipeline.fit(train_df["Review"], train_df["label"])

    preds = pipeline.predict(test_df["Review"])
    metrics = compute_classification_metrics(test_df["label"].tolist(), preds.tolist())

    print(f"TF-IDF + LR | Dogruluk: {metrics['accuracy']:.4f} | Makro-F1: {metrics['macro_f1']:.4f}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)
    return metrics


if __name__ == "__main__":
    train_and_evaluate()
