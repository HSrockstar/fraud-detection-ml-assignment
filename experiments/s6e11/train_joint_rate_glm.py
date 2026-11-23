"""One logistic classifier on cross-fitted value, digit and transaction-context rates."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from compare_tree_member import bootstrap_delta
from train_rate_glm import keys, rates


def context_keys(df):
    columns = keys(df)

    def add(left, right):
        a = pd.factorize(left, use_na_sentinel=False)[0]
        b = pd.factorize(right, use_na_sentinel=False)[0]
        code = pd.factorize(a * (int(b.max()) + 1) + b)[0]
        columns.append(code.astype(np.int32))

    for field in ("amount_usd", "account_balance_usd", "distance_from_home_km",
                  "hours_since_last_txn", "velocity_score", "merchant_risk_score"):
        whole = np.rint(100 * df[field]).astype(np.int64) // 100
        length = np.floor(np.log10(np.maximum(whole, 1))).astype(int) + 1
        lead = whole // (10 ** np.maximum(length - 2, 0))
        add(length, lead)
        add(length, np.rint(100 * df[field]).astype(int) % 100)
    risk = (df.merchant_risk_score // 5).astype(int)
    velocity = (df.velocity_score // 5).astype(int)
    amount = np.floor(4 * np.log1p(df.amount_usd)).astype(int)
    for field in ("cvv_retry_count", "prior_disputes", "auth_method", "used_vpn",
                  "is_foreign_transaction", "is_new_merchant", "ip_country_mismatch",
                  "billing_shipping_mismatch", "is_ai_generated_scam_attempt"):
        add(risk, df[field])
    add(risk, velocity)
    add(df.merchant_category, amount)
    add(df.channel, df.auth_method)
    add(df.is_foreign_transaction, df.ip_country_mismatch)
    return columns


def fit_and_score(cols, y, fit, score):
    x = np.empty((len(fit), len(cols)), dtype=np.float32)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(fit, y[fit]):
        x[b] = rates(cols, fit[a], fit[b], y)
    model = LogisticRegression(C=.1, max_iter=600)
    model.fit(x, y[fit])
    return model.decision_function(rates(cols, fit, score, y))


def main():
    root = Path(".kaggle-cache/competitions/playground-series-s6e11")
    tr = pd.read_csv(root / "train.csv")
    te = pd.read_csv(root / "test.csv")
    sample = pd.read_csv(root / "sample_submission.csv")
    assert sample.id.is_unique and sample.id.equals(te.id)
    folds = pd.read_csv("splits/seed42.csv").set_index("id").loc[tr.id, "fold"].to_numpy()
    y = tr.is_fraud.to_numpy(dtype=int)
    cols = context_keys(pd.concat([tr[te.columns], te], ignore_index=True).drop(columns="id"))
    dev = folds >= 0
    pred = np.full(len(y), np.nan)
    for f in range(5):
        valid = np.flatnonzero(folds == f)
        pred[valid] = fit_and_score(cols, y, np.flatnonzero(dev & (folds != f)), valid)
        print("fold", f, roc_auc_score(y[valid], pred[valid]), flush=True)
    with np.load("runs/self_rate_glm_oof.npz") as z:
        assert np.array_equal(z["id"], tr.id[dev])
        old = z["validation"]
    new = pred[dev]
    auc = roc_auc_score(y[dev], new)
    delta = auc - roc_auc_score(y[dev], old)
    fold_deltas = [roc_auc_score(y[dev][m], new[m])-roc_auc_score(y[dev][m], old[m])
                   for m in (folds[dev]==f for f in range(5))]
    print("OOF", auc, "delta", delta, "fold deltas", fold_deltas,
          "paired interval", bootstrap_delta(old, new, y[dev].astype(bool)), flush=True)
    np.savez_compressed("runs/self_joint_rate_glm_oof.npz", id=tr.id[dev], validation=new)
    logit = fit_and_score(cols, y, np.arange(len(y)), np.arange(len(y), len(y)+len(te)))
    assert np.isfinite(logit).all()
    sample.is_fraud = 1 / (1+np.exp(-logit.astype(np.float64)))
    sample.to_csv("runs/self_joint_rate_glm_submission.csv", index=False)
    print("saved single full-fit model predictions", len(sample), flush=True)


if __name__ == "__main__":
    main()
