"""Class-conditional mixtures of product distributions with hierarchical smoothing."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score

from compare_tree_member import bootstrap_delta
from train_context_network import AMOUNTS


def encode(table, fit):
    fields = []
    for name, s in table.items():
        # Allocate a separate unseen symbol within each coarse bucket.
        if name in AMOUNTS:
            s = np.rint(s * 100).astype(np.int64)
            edges = np.unique(np.quantile(s.iloc[fit], np.arange(1, 32) / 32))
            coarse = np.searchsorted(edges, s)
        else:
            coarse = np.zeros(len(s), dtype=int)
        groups = int(coarse.max()) + 1
        vocab = pd.Index(s.iloc[fit].unique())
        idx = vocab.get_indexer(s)
        code = np.where(idx < 0, coarse, idx + groups)
        parent = np.r_[np.arange(groups), coarse[s.index.get_indexer(s.iloc[fit].drop_duplicates().index)]]
        counts = np.bincount(code[fit], minlength=len(parent)) + .5
        reference = counts / np.bincount(parent, weights=counts)[parent]
        fields.append((code.astype(np.int32), parent, reference))
    return fields


def class_density(fields, fit, valid, k, seed):
    rng = np.random.default_rng(seed)
    # Anchor the initial responsibilities on one informative observed field.
    anchor = fields[0][0][fit]
    assignments = rng.integers(k, size=int(anchor.max()) + 1)[anchor]
    weights = .15 / k + .85 * (assignments[:, None] == np.arange(k))
    iterations = 1 if k == 1 else 15
    for i in range(iterations):
        masses = weights.sum(axis=0)
        rows = valid if i == iterations - 1 else fit
        score = np.broadcast_to(np.log(masses / masses.sum()), (len(rows), k)).copy()
        for code, parent, reference in fields:
            for h in range(k):
                counts = np.bincount(code[fit], weights=weights[:, h], minlength=len(parent))
                group_counts = np.bincount(parent, weights=counts)
                # Two normalized probability factors: bucket, then exact value given bucket.
                coarse = (group_counts + 1) / (masses[h] + len(group_counts))
                fine = (counts + 30 * reference) / (group_counts[parent] + 30)
                score[:, h] += np.log(coarse[parent] * fine)[code[rows]]
        normalizer = logsumexp(score, axis=1)
        if i + 1 < iterations:
            weights = np.exp(score - normalizer[:, None])
    return normalizer


def main():
    root = Path(".kaggle-cache/competitions/playground-series-s6e11")
    data = pd.read_csv(root / "train.csv")
    folds = pd.read_csv("splits/seed42.csv").set_index("id").loc[data.id, "fold"].to_numpy()
    y = data.is_fraud.to_numpy(int)
    table = data.drop(columns=["id", "is_fraud"])
    dest = Path("runs/latent_transactions")
    dest.mkdir(exist_ok=True)
    pred = {k: np.zeros(len(y)) for k in (1, 4)}
    for f in range(5):
        fit = np.flatnonzero((folds >= 0) & (folds != f))
        valid = np.flatnonzero(folds == f)
        fields = encode(table, fit)
        for k in pred:
            ll = [class_density(fields, fit[y[fit] == c], valid, k, 42+c) for c in (0, 1)]
            pred[k][valid] = ll[1] - ll[0] + np.log(y[fit].mean() / (1-y[fit].mean()))
            print(f, k, roc_auc_score(y[valid], pred[k][valid]), flush=True)
        np.savez_compressed(dest / f"fold{f}.npz", id=data.id.iloc[valid],
                            independent=pred[1][valid], latent=pred[4][valid])
    dev = folds >= 0
    for k in pred:
        print("OOF", k, roc_auc_score(y[dev], pred[k][dev]), flush=True)
    print("paired delta CI", bootstrap_delta(pred[1][dev], pred[4][dev], y[dev].astype(bool)), flush=True)


if __name__ == "__main__":
    main()
