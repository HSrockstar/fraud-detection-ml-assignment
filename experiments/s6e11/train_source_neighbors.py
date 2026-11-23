"""Retrieve source transactions by exact numeric anchors and score cross-field agreement."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from train_context_network import AMOUNTS, blocks, encoded_inputs, train
from compare_tree_member import bootstrap_delta


def retrieve(table, source):
    src, query = [], []
    for name in table.columns:
        vocab = pd.Index(source[name].unique())
        src.append(vocab.get_indexer(source[name]))
        query.append(vocab.get_indexer(table[name]))
    src = np.asarray(src, dtype=np.int32).T
    query = np.asarray(query, dtype=np.int32).T
    labels = source.is_fraud.to_numpy()
    result = []
    for anchor in AMOUNTS:
        j = table.columns.get_loc(anchor)
        order = np.argsort(src[:, j])
        code = src[order, j]
        start, end = np.searchsorted(code, query[:, j]), np.searchsorted(code, query[:, j], side="right")
        count = end - start
        cumulative = np.r_[0, labels[order].cumsum()]
        positives = cumulative[end] - cumulative[start]
        result.extend([np.log1p(count), np.log1p(positives)])
        if anchor in ("merchant_risk_score", "velocity_score"):
            continue
        # For selective anchors, score all matching source rows using the other fields.
        best = np.zeros((len(table), 2), dtype=np.float32)
        for offset in range(32):
            rows = np.flatnonzero((count > offset) & (count <= 32))
            if not len(rows):
                break
            ref = order[start[rows] + offset]
            agreement = (query[rows] == src[ref]).sum(axis=1) - 1
            cls = labels[ref].astype(int)
            best[rows, cls] = np.maximum(best[rows, cls], agreement)
        result.extend(best.T)
    return np.column_stack(result).astype(np.float32)


def main():
    root = Path(".kaggle-cache/competitions/playground-series-s6e11")
    table = pd.read_csv(root / "train.csv")
    folds = pd.read_csv("splits/seed42.csv").set_index("id").loc[table.id, "fold"].to_numpy()
    ids = table.id.to_numpy()
    y = table.is_fraud.to_numpy(dtype=int)
    table = table.drop(columns=["id", "is_fraud"])
    source = pd.read_csv("references/credit-card-fraud-detection-2026.zip")
    features = retrieve(table, source)
    money, context = blocks(table)
    out = Path("runs/source_neighbors")
    out.mkdir(exist_ok=True)
    previous, current = np.zeros(len(y)), np.zeros(len(y))
    for f in range(5):
        fit = np.flatnonzero((folds >= 0) & (folds != f))
        valid = np.flatnonzero(folds == f)
        x, v, boundary = encoded_inputs(table, y, fit, valid, money, context)
        mean, std = features[fit].mean(axis=0, dtype=np.float64), features[fit].std(axis=0, dtype=np.float64)
        def augment(tensor, rows):
            extra = np.clip((features[rows]-mean)/np.maximum(std,.05), -8, 8)
            return torch.cat((tensor, torch.as_tensor(extra, dtype=tensor.dtype, device=tensor.device)), dim=1)
        p = train(augment(x, fit), augment(v, valid), y[fit], boundary, "additive")
        with np.load(f"runs/context_network/fold{f}.npz") as z:
            assert np.array_equal(z["id"], ids[valid])
            previous[valid] = z["additive"]
        current[valid] = p
        print(f, roc_auc_score(y[valid], p), roc_auc_score(y[valid], p)-roc_auc_score(y[valid],previous[valid]), flush=True)
        np.savez_compressed(out / f"fold{f}.npz", id=ids[valid], source=p)
        del x, v
    dev = folds >= 0
    print("OOF", roc_auc_score(y[dev],current[dev]), "delta CI", bootstrap_delta(previous[dev], current[dev], y[dev].astype(bool)), flush=True)


if __name__ == "__main__":
    main()
