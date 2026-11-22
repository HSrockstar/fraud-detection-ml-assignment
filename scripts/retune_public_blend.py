"""把已提交的三成员融合改到更细的比例，并试一个上次没选中的成员。只看开发集。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud.common import ID, ROOT, TARGET, load_data, splits, timestamp, write_json

SOURCE = ROOT / "data/external"
SUBMITTED = ROOT / "submissions/public_oof_blend.csv"
OURS = "ours_7_3"
MEMBERS = (
    ("LightGBM 3 leaves, Part 5 recipe, 5 encoder seeds", "4188470eded4cdec"),
    ("LightGBM 4 leaves boosted from the joint GLM, full refit", "3d3b857be7461ed5"),
    ("XGBoost depth 2, Part 3 recipe, 5 encoder seeds, full refit", "82e9a79659bdf9d8"),
)


def rank01(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy() / (len(values) + 1.0)


def fast_auc(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score, kind="mergesort")
    ordered = y[order]
    n_pos = float(ordered.sum())
    n_neg = float(len(ordered) - n_pos)
    return float((ordered * np.cumsum(1.0 - ordered)).sum() / (n_pos * n_neg))


def compositions(slots: int, units: int):
    if slots == 1:
        yield (units,)
        return
    for count in range(units + 1):
        for rest in compositions(slots - 1, units - count):
            yield (count,) + rest


def search(y: np.ndarray, names: list[str], columns: list[np.ndarray], step: float) -> dict:
    units = int(round(1.0 / step))
    stacked = np.column_stack(columns)
    best_auc = -1.0
    best_weights = None
    for counts in compositions(len(columns), units):
        weights = np.asarray(counts, dtype=np.float64) / units
        auc = fast_auc(y, stacked @ weights)
        if auc > best_auc:
            best_auc = auc
            best_weights = weights
    confirmed = float(roc_auc_score(y, stacked @ best_weights))
    return {"members": names, "weights": [float(weight) for weight in best_weights], "dev_auc": confirmed, "step": step}


def main() -> None:
    folder = next(SOURCE.rglob("oof_manifest.json")).parent
    train, test = load_data()
    dev = splits(train)["partition"].eq("dev").to_numpy()
    y = train.loc[dev, TARGET].to_numpy()
    xgb = pd.read_parquet(ROOT / "artifacts/xgb_r2_parts_smooth80/oof.parquet")
    cat = pd.read_parquet(ROOT / "artifacts/cat_r2_parts/oof.parquet")
    ours = xgb.merge(cat, on=ID, suffixes=("_xgb", "_cat"))
    ours["probability"] = 0.7 * ours["probability_xgb"] + 0.3 * ours["probability_cat"]
    aligned = train.loc[dev, [ID]].merge(ours[[ID, "probability"]], on=ID, how="left")
    ours_test = pd.read_csv(ROOT / "submissions/xgb_cat_parts_70_30.csv")
    if not np.array_equal(ours_test[ID].to_numpy(), test[ID].to_numpy()):
        raise ValueError("当前 7:3 提交的 id 顺序与测试集不一致")
    dev_ranks = {OURS: rank01(aligned["probability"].to_numpy())}
    test_ranks = {OURS: rank01(ours_test[TARGET].to_numpy())}
    labels = {OURS: "我们的 7:3"}
    for name, key in MEMBERS:
        oof = np.load(folder / f"oof_{key}.npy")
        predicted = np.load(folder / f"test_{key}.npy")
        if len(oof) != len(train) or len(predicted) != len(test):
            raise ValueError(f"{name} 的长度与官方文件不一致")
        dev_ranks[name] = rank01(oof[dev])
        test_ranks[name] = rank01(predicted)
        labels[name] = name
    lgb3, lgb4, xgb2 = (name for name, _key in MEMBERS)
    groups = [
        search(y, [OURS, lgb4, lgb3], [dev_ranks[OURS], dev_ranks[lgb4], dev_ranks[lgb3]], 0.05),
        search(y, [OURS, lgb3, xgb2], [dev_ranks[OURS], dev_ranks[lgb3], dev_ranks[xgb2]], 0.05),
        search(y, [OURS, lgb4, xgb2], [dev_ranks[OURS], dev_ranks[lgb4], dev_ranks[xgb2]], 0.05),
        search(y, [OURS, lgb3, lgb4, xgb2], [dev_ranks[name] for name in (OURS, lgb3, lgb4, xgb2)], 0.1),
    ]
    submitted = pd.read_csv(SUBMITTED)
    if not np.array_equal(submitted[ID].to_numpy(), test[ID].to_numpy()):
        raise ValueError("已提交融合的 id 顺序与测试集不一致")
    submitted_rank = rank01(submitted[TARGET].to_numpy())
    for group in groups:
        blended = sum(weight * test_ranks[name] for weight, name in zip(group["weights"], group["members"]))
        group["spearman_vs_submitted"] = float(pd.Series(rank01(blended)).corr(pd.Series(submitted_rank), method="spearman"))
        group["test_blend"] = blended
    best = max(groups, key=lambda group: group["dev_auc"])
    distinct = [group for group in groups if group["spearman_vs_submitted"] < 0.998]
    chosen = max(distinct, key=lambda group: group["dev_auc"]) if distinct else best
    output = ROOT / "submissions/public_oof_blend_retune.csv"
    pd.DataFrame({ID: test[ID], TARGET: chosen["test_blend"]}).to_csv(output, index=False)
    summary = {
        "previous_weights": {"ours_7_3": 0.1, lgb4: 0.4, lgb3: 0.5},
        "previous_dev_auc": 0.9500308117905348,
        "previous_public_score": 0.95146,
        "groups": [{key: value for key, value in group.items() if key != "test_blend"} for group in groups],
        "chosen_members": chosen["members"],
        "chosen_weights": chosen["weights"],
        "chosen_dev_auc": chosen["dev_auc"],
        "chosen_spearman_vs_submitted": chosen["spearman_vs_submitted"],
        "file": "submissions/public_oof_blend_retune.csv",
        "created_at": timestamp(),
    }
    write_json(ROOT / "reports/public_oof_blend_retune.json", summary)
    print(json.dumps({
        "chosen_members": [labels[name] for name in chosen["members"]],
        "chosen_weights": chosen["weights"],
        "chosen_dev_auc": chosen["dev_auc"],
        "delta_vs_previous_dev": chosen["dev_auc"] - 0.9500308117905348,
        "spearman_vs_submitted": chosen["spearman_vs_submitted"],
        "all_dev_auc": [group["dev_auc"] for group in groups],
        "all_spearman": [group["spearman_vs_submitted"] for group in groups],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
