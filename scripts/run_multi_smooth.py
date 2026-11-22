"""深度 3 同时使用平滑 5、20、80 的目标编码。只在开发集上比较。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud.common import ID, ROOT, TARGET, write_json
from fraud.experiments import run_cv

REFERENCE_BLEND = 0.9478163587782216
REFERENCE_XGB = "xgb_r2_parts_smooth80"
NAME = "xgb_r3_parts_smooth_multi"
PARAMS = {
    "learning_rate": 0.0886816342974295,
    "max_depth": 3,
    "min_child_weight": 18.02471215771192,
    "subsample": 0.9974913605156784,
    "colsample_bytree": 0.9996687449838664,
    "reg_lambda": 1.1500035567289912,
    "reg_alpha": 2.7305861513999963,
    "n_estimators": 2000,
    "max_bin": 256,
    "target_encoding": True,
    "target_smooth": [5.0, 20.0, 80.0],
    "value_encoding": "parts",
}


def main() -> None:
    result = run_cv(NAME, "xgb", PARAMS, after_freeze=True)
    devices = {fold["training_device"] for fold in result["folds"]}
    if any(not str(device).startswith("cuda") for device in devices):
        raise RuntimeError(f"未在 CUDA 上训练: {devices}")
    reference = json.loads((ROOT / "reports/experiments" / f"{REFERENCE_XGB}.json").read_text(encoding="utf-8"))
    new = pd.read_parquet(ROOT / "artifacts" / NAME / "oof.parquet")
    old = pd.read_parquet(ROOT / "artifacts" / REFERENCE_XGB / "oof.parquet")
    cat = pd.read_parquet(ROOT / "artifacts/cat_r2_parts/oof.parquet")
    merged = new.merge(old, on=ID, suffixes=("_new", "_old")).merge(cat, on=ID)
    y = merged[f"{TARGET}_new"].to_numpy()
    if not np.array_equal(y, merged[f"{TARGET}_old"].to_numpy()) or not np.array_equal(y, merged[TARGET].to_numpy()):
        raise ValueError("折外标签不一致")
    blend = 0.7 * merged["probability_old"].to_numpy() + 0.3 * merged["probability"].to_numpy()
    new_p = merged["probability_new"].to_numpy()
    spearman_blend = float(pd.Series(new_p).corr(pd.Series(blend), method="spearman"))
    auc = result["oof_default"]["roc_auc"]
    reference_folds = [fold["metrics"]["roc_auc"] for fold in reference["folds"]]
    new_folds = [fold["metrics"]["roc_auc"] for fold in result["folds"]]
    summary = {
        "name": NAME,
        "smooth": [5.0, 20.0, 80.0],
        "oof_auc": auc,
        "oof_ap": result["oof_default"]["average_precision"],
        "fold_auc": new_folds,
        "reference_fold_auc": reference_folds,
        "fold_deltas_vs_depth3": [left - right for left, right in zip(new_folds, reference_folds)],
        "iterations": [fold["selected_iterations"] for fold in result["folds"]],
        "seconds": result["total_seconds"],
        "depth3_oof_auc": reference["oof_default"]["roc_auc"],
        "delta_vs_depth3": auc - reference["oof_default"]["roc_auc"],
        "blend_7_3_oof_auc": REFERENCE_BLEND,
        "blend_7_3_recomputed": float(roc_auc_score(y, blend)),
        "delta_vs_blend": auc - REFERENCE_BLEND,
        "spearman_vs_blend": spearman_blend,
        "improved": auc >= REFERENCE_BLEND + 0.001 and spearman_blend < 0.98 and all(delta > 0 for delta in [left - right for left, right in zip(new_folds, reference_folds)]),
    }
    write_json(ROOT / "reports/multi_smooth.json", summary)
    print(json.dumps({
        "oof_auc": auc,
        "delta_vs_depth3": summary["delta_vs_depth3"],
        "delta_vs_blend": summary["delta_vs_blend"],
        "spearman_vs_blend": spearman_blend,
        "fold_deltas_vs_depth3": summary["fold_deltas_vs_depth3"],
        "improved": summary["improved"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
