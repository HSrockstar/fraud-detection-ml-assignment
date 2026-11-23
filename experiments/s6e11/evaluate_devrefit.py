"""One holdout check of the fixed equal-weight fold/refit inference rule."""
from pathlib import Path
import numpy as np
import pandas as pd
from compare_tree_member import placements, rank


def main():
    data = Path(".kaggle-cache/competitions/playground-series-s6e11")
    splits = pd.read_csv("splits/seed42.csv").set_index("id")
    train = pd.read_csv(data / "train.csv", usecols=["id", "is_fraud"])
    hold = train.loc[splits.loc[train.id, "fold"].to_numpy() == -1]
    y = hold.is_fraud.to_numpy() == 1
    means, refits = [], []
    for seed in (42, 43):
        prefix = Path("runs") / f"realmlp-e4-s{seed}"
        folds = []
        for fold in range(5):
            with np.load(f"{prefix}-f{fold}.npz") as z:
                assert np.array_equal(z["holdout_id"], hold.id)
                folds.append(z["holdout"])
        means.append(rank(np.mean(folds, axis=0, dtype=np.float64)))
        with np.load(f"{prefix}-devrefit.npz") as z:
            assert np.array_equal(z["holdout_id"], hold.id)
            assert np.isfinite(z["holdout"]).all()
            refits.append(rank(z["holdout"]))
    avg = rank(np.mean(means, axis=0))
    refit = rank(np.mean(refits, axis=0))
    p0, n0 = placements(avg, y)
    for name, x in [("fold mean", avg), ("refit", refit), ("50/50", (avg+refit)/2)]:
        p, n = placements(x, y)
        delta = p.mean()-p0.mean()
        se = np.sqrt(np.var(p-p0,ddof=1)/len(p) + np.var(n-n0,ddof=1)/len(n))
        print(f"{name}: H AUC {p.mean():.7f}, delta {delta:+.7f}, 95% CI [{delta-1.96*se:+.7f}, {delta+1.96*se:+.7f}]")


if __name__ == "__main__":
    main()
