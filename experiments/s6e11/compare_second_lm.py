"""Replace 5 or 10 percentage points of megayak LM, or raise tree weight to 15%."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from compare_tree_member import bootstrap_delta, load_comparison_data, placements, public_oof, rank


def main():
    base, tree, tree_test, sample, y, split, mask = load_comparison_data()
    original = .9 * base + .1 * tree
    scored = pd.read_csv("runs/tree10_lm_mlp_fullrefit50.csv")
    assert scored.id.equals(sample.id)
    directory = Path(".kaggle-cache/datasets/masayakawamata/s6e11-gpt2-target-first-oof/versions/2")
    manifest = json.loads((directory / "manifest.json").read_text())
    data = Path(".kaggle-cache/competitions/playground-series-s6e11")
    for key, parent in [("train_csv", data), ("test_csv", data), ("fold_ids", directory)]:
        info = manifest["inputs"][key]
        assert sha256((parent / info["file"]).read_bytes()).hexdigest() == info["sha256"]
    raw = np.load(directory / "oof.npy")
    raw_test = np.load(directory / "test.npy")
    assert raw.shape == mask.shape and raw_test.shape == (len(sample),)
    assert np.isfinite(raw).all() and np.isfinite(raw_test).all()
    new = rank(raw[mask])
    new_test = rank(raw_test)
    old = public_oof("s6e11-target-first-lm", "lm_oof", mask)
    with ZipFile("references/s6e11-target-first-lm.zip") as z:
        old_test = rank(np.load(BytesIO(z.read("lm_test.npy"))))
    print("LM rank correlation", np.corrcoef(old,new)[0,1], flush=True)
    plans = [(f"lm2_{int(w*100)}", original+w*(new-old), scored.is_fraud+w*(new_test-old_test)) for w in (.05,.1)]
    plans.append(("tree15", .85*base+.15*tree, (.85*scored.is_fraud+.05*tree_test)/.9))
    auc = lambda x, labels: placements(x, labels)[0].mean()
    baseline = auc(original, y)
    folds = split[mask]
    report = []
    for name, pred, test_pred in plans:
        score = auc(pred,y)
        deltas = [auc(pred[folds==f], y[folds==f])-auc(original[folds==f], y[folds==f]) for f in range(5)]
        lo, hi = bootstrap_delta(original, pred, y)
        row = dict(candidate=name, baseline_auc=baseline, auc=score, delta=score-baseline,
                   ci_low=lo, ci_high=hi, **{f"delta_fold_{f}":d for f,d in enumerate(deltas)})
        report.append(row)
        print(name, score, score-baseline, lo, hi, [round(float(d),7) for d in deltas], flush=True)
        out = scored.assign(is_fraud=test_pred)
        assert out.is_fraud.between(0,1).all() and out.id.is_unique
        out.to_csv(f"runs/fullrefit_{name}.csv",index=False)
    pd.DataFrame(report).to_csv("runs/second_lm_tree15_results.csv",index=False)


if __name__ == "__main__":
    main()
