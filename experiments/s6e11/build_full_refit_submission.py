"""Replace only the 4.5% development-refit contribution in the scored submission."""
from pathlib import Path

import numpy as np
import pandas as pd

from compare_tree_member import rank


def seed_average(scope, ids):
    predictions = []
    for seed in (42, 43):
        with np.load(Path("runs") / f"realmlp-e4-s{seed}-{scope}.npz") as data:
            values = data["test"]
            assert np.array_equal(data["test_id"], ids)
            assert values.shape == ids.shape and np.isfinite(values).all()
            predictions.append(rank(values))
    return rank(np.mean(predictions, axis=0))


if __name__ == "__main__":
    root = Path("runs")
    previous = pd.read_csv(root / "tree10_lm_mlp_refit50.csv")
    sample = pd.read_csv(".kaggle-cache/competitions/playground-series-s6e11/sample_submission.csv")
    assert sample.id.is_unique and previous.id.equals(sample.id)
    assert previous.columns.equals(sample.columns) and len(previous) == 299984
    old = seed_average("devrefit", sample.id.to_numpy())
    new = seed_average("fullrefit", sample.id.to_numpy())
    out = previous.assign(is_fraud=previous.is_fraud + 0.045 * (new - old))
    assert out.is_fraud.between(0, 1).all()
    path = root / "tree10_lm_mlp_fullrefit50.csv"
    out.to_csv(path, index=False)
    print(path, len(out), "rows")
