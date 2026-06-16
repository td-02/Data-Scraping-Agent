#!/usr/bin/env python3
"""Train a logistic scoring model for business suitability outcomes."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ["rating", "review_count", "competitor_count", "avg_competitor_rating"]
LABEL_COLUMN = "outcome_label"
DEFAULT_MODEL_PATH = Path("models") / "scoring_model.pkl"


def train_model(input_csv: Path, output_path: Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Fit a logistic regression model and save it to disk."""
    df = pd.read_csv(input_csv)
    missing = [column for column in [*FEATURE_COLUMNS, LABEL_COLUMN] if column not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {', '.join(missing)}")

    x = df[FEATURE_COLUMNS].fillna(0.0)
    y = df[LABEL_COLUMN].astype(int)
    if y.nunique() < 2:
        raise ValueError("outcome_label must contain both 0 and 1 examples")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logistic_regression", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(x, y)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    return model


def main() -> None:
    """CLI entrypoint for training the optional scoring model."""
    parser = argparse.ArgumentParser(description="Train a business suitability scoring model.")
    parser.add_argument("input_csv", type=Path, help="CSV with rating/review/competition features and outcome_label")
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH, help="Output model path")
    args = parser.parse_args()

    train_model(args.input_csv, args.out)
    print(f"Saved scoring model to {args.out}")


if __name__ == "__main__":
    main()
