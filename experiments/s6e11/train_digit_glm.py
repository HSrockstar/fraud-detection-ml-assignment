"""Development-only additive logistic regression with fixed categorical hashing."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

from compare_tree_member import bootstrap_delta, load_comparison_data, rank


def design(table):
    # A separate fixed hash space for each column needs no fitted vocabulary.
    blocks = []
    width = 4096

    def add(values):
        hashes = pd.util.hash_pandas_object(values, index=False).to_numpy()
        blocks.append((hashes % width).astype(np.int32) + width * len(blocks))

    numeric = table.select_dtypes(include="number")
    decimals = [c for c in numeric if (numeric[c] % 1 != 0).any()]
    for name, col in table.items():
        if name in decimals:
            add(np.floor(10 * np.log1p(col)).astype(int))
        else:
            add(col)
    base_count = len(blocks)
    for name in decimals:
        cents = (100 * table[name]).round().astype(int)
        for values in (cents, cents // 100, cents % 100, cents // 1000, cents // 10 % 10):
            add(values)

    def matrix(features):
        ids = np.column_stack(features).ravel()
        offsets = np.arange(0, len(ids)+1, len(features), dtype=np.int32)
        return csr_matrix((np.ones(len(ids), dtype=np.float32), ids, offsets),
                          shape=(len(table), width*len(features)))
    return matrix(blocks[:base_count]), matrix(blocks)


def main():
    train = pd.read_csv(".kaggle-cache/competitions/playground-series-s6e11/train.csv")
    split = pd.read_csv("splits/seed42.csv").set_index("id").loc[train.id, "fold"].to_numpy()
    train = train.loc[split >= 0].reset_index(drop=True)
    folds = split[split >= 0]
    labels = train.is_fraud.to_numpy(dtype=int)
    raw, digits = design(train.drop(columns=["id", "is_fraud"]))
    results = {}
    for name, features in [("coarse",raw), ("digits",digits)]:
        predictions = np.empty(len(train))
        for fold in range(5):
            a = folds != fold
            model = SGDClassifier(loss="log_loss", alpha=0.00001, max_iter=16,
                                  tol=None, average=True, random_state=42)
            model.fit(features[a], labels[a])
            predictions[~a] = model.decision_function(features[~a])
        results[name] = predictions
        print(name, roc_auc_score(labels,predictions), flush=True)
    out = Path("runs/self_digit_glm_oof.npz")
    np.savez_compressed(out, id=train.id.to_numpy(), **results)
    mlp = []
    for seed in (42,43):
        rows = []
        for f in range(5):
            with np.load(f"runs/realmlp-e4-s{seed}-f{f}.npz") as z:
                rows.append(pd.Series(z["validation"], index=z["validation_id"]))
        mlp.append(rank(pd.concat(rows, verify_integrity=True).loc[train.id]))
    mlp = rank(np.mean(mlp,axis=0))
    pool, tree, _, _, _, _, _ = load_comparison_data()
    for name, base, w in [("self MLP",mlp,.5),("public blend OOF proxy",.9*pool+.1*tree,.1)]:
        pred = (1-w)*base+w*rank(results["digits"])
        delta = lambda m: roc_auc_score(labels[m],pred[m])-roc_auc_score(labels[m],base[m])
        print(name, "baseline",roc_auc_score(labels,base),"blend",roc_auc_score(labels,pred),
              "fold deltas",[round(delta(folds==f),7) for f in range(5)],
              "paired interval",bootstrap_delta(base,pred,labels.astype(bool)),flush=True)


if __name__ == "__main__":
    main()
