"""Match the public LM fold protocol; report ranking errors only on the original development IDs."""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from compare_tree_member import placements, bootstrap_delta
from train_context_network import blocks, encoded_inputs, train


def main():
    frame = pd.read_csv(".kaggle-cache/competitions/playground-series-s6e11/train.csv")
    ids = frame.id.to_numpy()
    assert frame.id.is_unique and np.all(np.diff(ids) == 1)
    y = frame.is_fraud.to_numpy(int)
    dev = pd.read_csv("splits/seed42.csv").set_index("id").loc[ids, "fold"].to_numpy() >= 0
    root = Path("runs/lm_error_audit")
    root.mkdir(exist_ok=True)
    with ZipFile("references/s6e11-target-first-lm.zip") as archive:
        lm = np.load(BytesIO(archive.read("lm_oof.npy"))).reshape(-1)
    assert len(lm) == len(ids) and np.isfinite(lm).all()
    table = frame.drop(columns=["is_fraud", "id"])
    money, context = blocks(table)
    output = np.full(len(y), np.nan)
    for fold, (fit, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(ids, y)):
        valid = valid[dev[valid]]
        x, v, boundary = encoded_inputs(table, y, fit, valid, money, context)
        output[valid] = train(x, v, y[fit], boundary, "additive")
        print(fold, len(fit), roc_auc_score(y[valid], output[valid]), roc_auc_score(y[valid], lm[valid]), flush=True)
        del x, v
    d = frame.loc[dev].copy()
    label = y[dev].astype(bool)
    base, candidate = output[dev], lm[dev]
    bp, _ = placements(base, label)
    lp, _ = placements(candidate, label)
    print("AUCs", bp.mean(), lp.mean(), "paired CI", bootstrap_delta(base, candidate, label), flush=True)
    fraud = d.loc[label].copy()
    fraud["base_placement"] = bp
    fraud["lm_placement"] = lp
    fraud["delta"] = lp - bp
    fraud.to_csv(root / "fraud_placements.csv", index=False)
    np.savez_compressed(root / "aligned.npz", id=ids[dev], additive=base, lm=candidate)
    print("frauds", len(fraud), "positive and negative contributions", np.maximum(lp-bp,0).mean(), np.minimum(lp-bp,0).mean())
    for name in ["amount_usd","account_balance_usd","merchant_risk_score","velocity_score",
                 "cvv_retry_count","prior_disputes","channel","is_ai_generated_scam_attempt"]:
        group = pd.qcut(d[name], 4, duplicates="drop")[label] if name in ["amount_usd","account_balance_usd","merchant_risk_score","velocity_score"] else fraud[name]
        print(name, fraud.groupby(group, observed=True).delta.agg(["size","mean","sum"]).round(5).to_string())
    print("largest changes",fraud.nlargest(3,"delta")[["id","base_placement","lm_placement"]].to_dict("records"),
          fraud.nsmallest(3,"delta")[["id","base_placement","lm_placement"]].to_dict("records"))


if __name__ == "__main__":
    main()
