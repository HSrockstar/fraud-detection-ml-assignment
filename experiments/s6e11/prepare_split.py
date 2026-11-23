"""Create the fixed S6E11 development folds and holdout, keyed by competition ID."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / ".kaggle-cache" / "competitions" / "playground-series-s6e11"
OUT = ROOT / "splits" / "seed42.csv"


def main():
    train = pd.read_csv(DATA / "train.csv", usecols=["id", "is_fraud"])
    test = pd.read_csv(DATA / "test.csv", usecols=["id"])
    sample = pd.read_csv(DATA / "sample_submission.csv")
    assert train.id.is_unique and test.id.is_unique
    assert not train.id.isin(test.id).any() and test.id.equals(sample.id)
    assert set(train.is_fraud.unique()) == {0, 1}

    # Canonical order makes the split independent of CSV row order.
    train = train.sort_values("id", ignore_index=True)
    rng = np.random.Generator(np.random.PCG64(42))
    folds = np.full(len(train), -1, dtype=np.int8)
    for label in (0, 1):
        indices = rng.permutation(np.flatnonzero(train.is_fraud == label))
        development = indices[round(len(indices) / 5):]
        for fold, rows in enumerate(np.array_split(development, 5)):
            folds[rows] = fold

    assignments = train[["id"]].assign(fold=folds)
    OUT.parent.mkdir(exist_ok=True)
    assignments.to_csv(OUT, index=False)
    print(f"train={len(train)}, test={len(test)}, positive={train.is_fraud.sum()}")
    print(assignments.fold.value_counts().sort_index())


if __name__ == "__main__":
    main()
