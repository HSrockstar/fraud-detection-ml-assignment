"""Compare one external LightGBM with the submitted 80/10/10 blend on development OOF."""
import hashlib
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


def rank(values):
    return pd.Series(values).rank(pct=True).to_numpy()


def placements(x, y):
    pos, neg = x[y], x[~y]
    def mid(reference, values):
        ordered = np.sort(reference)
        return (ordered.searchsorted(values, "left") + ordered.searchsorted(values, "right")) / (2 * len(ordered))
    return mid(neg, pos), 1 - mid(pos, neg)


def public_oof(slug, member, mask):
    with ZipFile(Path("references") / (slug + ".zip")) as z:
        return rank(np.load(BytesIO(z.read(member + ".npy")))[mask])


def bootstrap_delta(base, candidate, y, repeats=500):
    """Stratified paired bootstrap, including half credit for tied scores."""
    positive, negative = np.flatnonzero(y), np.flatnonzero(~y)
    orders = []
    for scores in (base, candidate):
        order = np.argsort(scores[negative])
        ordered = scores[negative][order]
        orders.append((order, ordered.searchsorted(scores[positive], "left"),
                       ordered.searchsorted(scores[positive], "right")))
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(repeats):
        sampled_pos = rng.integers(len(positive), size=len(positive))
        neg_count = np.bincount(rng.integers(len(negative), size=len(negative)), minlength=len(negative))
        auc = []
        for order, left, right in orders:
            cumulative = np.r_[0, np.cumsum(neg_count[order])]
            auc.append((cumulative[left[sampled_pos]] + cumulative[right[sampled_pos]]).mean() / (2*len(negative)))
        deltas.append(auc[1]-auc[0])
    return np.quantile(deltas, [0.025, 0.975])


def load_comparison_data():
    root = Path(".kaggle-cache/competitions/playground-series-s6e11")
    train = pd.read_csv(root / "train.csv", usecols=["id", "is_fraud"])
    split = pd.read_csv("splits/seed42.csv").set_index("id").loc[train.id, "fold"].to_numpy()
    mask = split >= 0
    dev = train.loc[mask]
    y = dev.is_fraud.to_numpy() == 1
    seeds = []
    for seed in (42, 43):
        rows = []
        for fold in range(5):
            with np.load(f"runs/realmlp-e4-s{seed}-f{fold}.npz") as z:
                rows.append(pd.Series(z["validation"], index=z["validation_id"]))
        oof = pd.concat(rows, verify_integrity=True)
        assert len(oof) == len(dev)
        seeds.append(rank(oof.loc[dev.id]))
    base = (0.8 * public_oof("s6e11-target-first-lm", "lm_oof", mask)
            + 0.1 * public_oof("s6e11-realmlp-member", "mlp4_oof", mask)
            + 0.1 * rank(np.mean(seeds, axis=0)))
    bank = Path(".kaggle-cache/datasets/wguesdon/s6e11-tree-ensemble-predictions/versions/3")
    name = "encoded_lightgbm.npz"
    assert hashlib.sha256((bank / name).read_bytes()).hexdigest() == json.loads((bank / "manifest.json").read_text())["files"][name]
    current = pd.read_csv("runs/candidate_lm80_self4_10_public4_10.csv")
    with np.load(bank / name) as z:
        assert np.array_equal(z["train_id"], train.id) and np.array_equal(z["test_id"], current.id)
        tree = rank(z["oof"][mask])
        tree_test = rank(z["fold_test"].mean(axis=0, dtype=np.float64))
    assert np.isfinite(tree).all() and np.isfinite(tree_test).all()
    return base, tree, tree_test, current, y, split, mask


def main():
    base, tree, tree_test, current, y, split, mask = load_comparison_data()
    pos0, neg0 = placements(base, y)
    print("baseline", pos0.mean())
    for w in (0.05, 0.10, 0.20):
        x = (1 - w) * base + w * tree
        pos, neg = placements(x, y)
        diff = pos.mean() - pos0.mean()
        se = np.sqrt(np.var(pos-pos0, ddof=1)/len(pos) + np.var(neg-neg0, ddof=1)/len(neg))
        deltas = []
        for f in range(5):
            rows = split[mask] == f
            deltas.append(placements(x[rows], y[rows])[0].mean() - placements(base[rows], y[rows])[0].mean())
        print(f"{w:.0%}: AUC {pos.mean():.7f}, delta {diff:+.7f}, paired 95% CI [{diff-1.96*se:+.7f}, {diff+1.96*se:+.7f}], folds", [round(float(v), 7) for v in deltas])
        if w == 0.10:
            print("10% paired stratified bootstrap 95% interval", bootstrap_delta(base, x, y))
        out = current.assign(is_fraud=(1-w)*current.is_fraud + w*tree_test)
        assert out.is_fraud.between(0, 1).all()
        out.to_csv(f"runs/tree{int(100*w):02d}_lm_mlp.csv", index=False)


if __name__ == "__main__":
    main()
