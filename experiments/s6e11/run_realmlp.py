"""Run the megayak RealMLP configuration on our fixed development folds."""
import gc
import io
import argparse
from contextlib import redirect_stdout
import numpy as np, pandas as pd
from pathlib import Path
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder
from pytabkit import RealMLP_TD_Classifier

parser = argparse.ArgumentParser()
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--refit", action="store_true", help="Fit the selected 4-epoch model on all development rows")
mode.add_argument("--full-refit", action="store_true", help="Fit the selected model on all labelled rows for submission")
args = parser.parse_args()

D = Path(".kaggle-cache/competitions/playground-series-s6e11")
tr = pd.read_csv(D / "train.csv")
te = pd.read_csv(D / "test.csv")
split = pd.read_csv("splits/seed42.csv").set_index("id").loc[tr.id, "fold"].to_numpy()
hold = split == -1
if args.full_refit:
    hold = np.zeros(len(tr), dtype=bool)
dev = ~hold
dev_ids = tr.loc[dev, "id"].to_numpy()
hold_ids = tr.loc[hold, "id"].to_numpy()
test_ids = te.id.to_numpy()
folds = split[dev]
y = tr.loc[dev, "is_fraud"].astype(int).to_numpy()
held_features = tr.loc[hold].drop(columns=["id", "is_fraud"])
te = pd.concat([held_features, te.drop(columns="id")], ignore_index=True)
hold_size = len(held_features)
tr = tr.loc[dev].drop(columns=["id", "is_fraud"]).reset_index(drop=True)
assert torch.cuda.is_available()
print("GPU", torch.cuda.get_device_name(), flush=True)

cat = [c for c in tr.columns if tr[c].dtype == bool or not pd.api.types.is_numeric_dtype(tr[c])]
num = [c for c in tr.columns if c not in cat]
dec = [c for c in num if (tr[c] % 1 != 0).any()]
print("decimal columns", dec, flush=True)


def coded(a, b):
    codes, uniq = pd.factorize(a)
    look = {u: i for i, u in enumerate(uniq)}
    return codes.astype("int32"), b.map(look).fillna(-1).astype("int32")


new = []
for c in num:
    tr[c + "_int"], te[c + "_int"] = coded(np.floor(tr[c]), np.floor(te[c])); new.append(c + "_int")
for c in dec:
    tr[c + "_cents"], te[c + "_cents"] = coded((tr[c] * 100).round() % 100, (te[c] * 100).round() % 100); new.append(c + "_cents")
for c, nb in [("merchant_risk_score", 7), ("hours_since_last_txn", 500), ("hours_since_last_txn", 1000)]:
    kb = KBinsDiscretizer(n_bins=nb, encode="ordinal", strategy="quantile", subsample=None)
    n = f"{c}_q{nb}"
    tr[n] = kb.fit_transform(tr[[c]]).ravel().astype("int32"); te[n] = kb.transform(te[[c]]).ravel().astype("int32"); new.append(n)
crosses = []
for cols in [("merchant_category", "card_type", "txn_count_last_24h"), ("merchant_category", "customer_age")]:
    n = "x_" + "_".join(cols)
    def j(d):
        key = d[cols[0]].astype(str)
        for col in cols[1:]:
            key = key.str.cat(d[col].astype(str), sep="_")
        return key
    tr[n], te[n] = coded(j(tr), j(te)); new.append(n); crosses.append(n)
for n in new:
    tr[n] = tr[n].astype("category"); te[n] = te[n].astype("category")
CATS = cat + new
TE_SRC = crosses + ["hours_since_last_txn"]
TE_NAMES = [c + "_te" for c in TE_SRC]
print("inputs", tr.shape[1] + len(TE_NAMES), "categorical", len(CATS), flush=True)

P = dict(verbosity=0, val_metric_name="1-auc_ovr", n_ens=8, n_epochs=2, stop_epoch=2, batch_size=288, use_early_stopping=False,
         lr=0.053, wd=0.018, sq_mom=0.988, lr_sched="flat_anneal", wd_sched="cos_log_15", first_layer_lr_factor=0.25,
         embedding_size=5, max_one_hot_cat_size=16, hidden_sizes=[512, 256, 128], act="silu", p_drop=0.05, p_drop_sched="pow5",
         plr_hidden_1=16, plr_hidden_2=8, plr_act_name="gelu", plr_lr_factor=0.1151, plr_sigma=0.75, ls_eps=0.01,
         ls_eps_sched="sqrt_cos", add_front_scale=False, bias_init_mode="neg-uniform-dynamic-2",
         tfms=["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding", "l2_normalize"])


def fit_predict(a, b, seed):
    """Train on a, then score the development fold, holdout and test."""
    Xa, Xt = tr.iloc[a].copy(), te.copy()
    enc = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=seed)
    Xa[TE_NAMES] = enc.fit_transform(Xa[TE_SRC], y[a]); Xt[TE_NAMES] = enc.transform(Xt[TE_SRC])
    model = RealMLP_TD_Classifier(**{**P, "random_state": seed})
    with redirect_stdout(io.StringIO()):
        # Fixed stop_epoch. The API's monitoring rows are from the fit set.
        model.fit(Xa, y[a], Xa.iloc[:256], y[a[:256]], cat_col_names=CATS)
    pb = None
    if b is not None:
        Xb = tr.iloc[b].copy()
        Xb[TE_NAMES] = enc.transform(Xb[TE_SRC])
        pb = model.predict_proba(Xb)[:, 1]
    pt = model.predict_proba(Xt)[:, 1]
    del model; gc.collect(); torch.cuda.empty_cache()
    return pb, pt


if args.refit or args.full_refit:
    P.update(n_epochs=4, stop_epoch=4, device="cuda")
    scope = "fullrefit" if args.full_refit else "devrefit"
    for seed in (42, 43):
        path = Path("runs") / f"realmlp-e4-s{seed}-{scope}.npz"
        if path.exists():
            continue
        print("start", scope, seed, "rows", len(y), flush=True)
        _, pt = fit_predict(np.arange(len(y)), None, seed)
        if not np.isfinite(pt).all():
            raise ValueError("Non-finite refit predictions")
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, holdout=pt[:hold_size], test=pt[hold_size:], holdout_id=hold_ids, test_id=test_ids)
        tmp.replace(path)
        print("saved", path, flush=True)
    raise SystemExit(0)

for epochs in (2, 4):
    P.update(n_epochs=epochs, stop_epoch=epochs, device="cuda")
    for seed in (42, 43):
        for fold in range(5):
            path = Path("runs") / f"realmlp-e{epochs}-s{seed}-f{fold}.npz"
            path.parent.mkdir(exist_ok=True)
            if path.is_file():
                continue
            a, b = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            print("start", epochs, seed, fold, flush=True)
            pb, pt = fit_predict(a, b, seed)
            if not (np.isfinite(pb).all() and np.isfinite(pt).all()):
                raise ValueError("Non-finite predictions")
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, validation=pb, holdout=pt[:hold_size], test=pt[hold_size:],
                     validation_id=dev_ids[b], holdout_id=hold_ids, test_id=test_ids)
            tmp.replace(path)
            print(path, "development fold AUC", roc_auc_score(y[b], pb), flush=True)
