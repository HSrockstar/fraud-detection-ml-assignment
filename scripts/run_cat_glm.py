"""给整数档和小数档上的 CatBoost 加上折内逻辑回归 logit 起点。只比较开发集。"""
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
from fraud.models import fit_model, predict

REFERENCE_BLEND = 0.9478163587782216
NAME = "cat_r3_glm_parts"
PARAMS = {
    "iterations": 2000,
    "depth": 6,
    "learning_rate": 0.04,
    "l2_leaf_reg": 8.0,
    "border_count": 128,
    "value_encoding": "parts",
    "target_encoding": True,
    "target_smooth": 80.0,
    "glm_boost": True,
    "glm_C": 0.01,
}


def assert_baseline_changes_catboost() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "amount": rng.normal(size=240),
        "category": np.tile(["A", "B", "C"], 80),
    })
    y = np.tile([0, 1], 120)
    shared = {"iterations": 20, "depth": 2, "learning_rate": 0.2, "target_encoding": True, "target_smooth": 80.0}
    plain = fit_model("cat", shared, frame, y, seed=0, fixed_iterations=12)
    boosted = fit_model("cat", {**shared, "glm_boost": True, "glm_C": 0.01}, frame, y, seed=0, fixed_iterations=12)
    if plain["training_device"] != "GPU" or boosted["training_device"] != "GPU":
        raise RuntimeError(f"CatBoost 未在 GPU 上训练: {plain['training_device']}, {boosted['training_device']}")
    plain_p = predict(plain, frame)
    boosted_p = predict(boosted, frame)
    if not np.isfinite(boosted_p).all() or np.allclose(plain_p, boosted_p):
        raise RuntimeError("CatBoost 的逻辑回归起点没有改变预测")


def fold_aucs(y, probability, folds) -> list[float]:
    return [float(roc_auc_score(y[folds == fold], probability[folds == fold])) for fold in range(3)]


def main() -> None:
    assert_baseline_changes_catboost()
    result = run_cv(NAME, "cat", PARAMS, after_freeze=True)
    devices = {fold["training_device"] for fold in result["folds"]}
    if devices != {"GPU"}:
        raise RuntimeError(f"未在 GPU 上训练: {devices}")
    new = pd.read_parquet(ROOT / "artifacts" / NAME / "oof.parquet")
    old = pd.read_parquet(ROOT / "artifacts/cat_r2_parts/oof.parquet")
    xgb = pd.read_parquet(ROOT / "artifacts/xgb_r2_parts_smooth80/oof.parquet")
    merged = new.merge(old, on=ID, suffixes=("_new", "_old")).merge(xgb, on=ID)
    y = merged[f"{TARGET}_new"].to_numpy()
    if not np.array_equal(y, merged[f"{TARGET}_old"].to_numpy()) or not np.array_equal(y, merged[TARGET].to_numpy()):
        raise ValueError("折外标签不一致")
    folds = merged["fold_new"].to_numpy()
    if not np.array_equal(folds, merged["fold_old"].to_numpy()) or not np.array_equal(folds, merged["fold"].to_numpy()):
        raise ValueError("折编号不一致")
    new_p = merged["probability_new"].to_numpy()
    old_p = merged["probability_old"].to_numpy()
    xgb_p = merged["probability"].to_numpy()
    reference = 0.7 * xgb_p + 0.3 * old_p
    rows = []
    for step in range(11):
        weight = step / 10
        blended = weight * xgb_p + (1 - weight) * new_p
        rows.append({
            "xgb_weight": weight,
            "cat_weight": 1 - weight,
            "oof_auc": float(roc_auc_score(y, blended)),
            "fold_auc": fold_aucs(y, blended, folds),
        })
    best = max(rows, key=lambda row: row["oof_auc"])
    best_p = best["xgb_weight"] * xgb_p + best["cat_weight"] * new_p
    reference_folds = fold_aucs(y, reference, folds)
    fold_deltas = [left - right for left, right in zip(best["fold_auc"], reference_folds)]
    summary = {
        "name": NAME,
        "cat_oof_auc": result["oof_default"]["roc_auc"],
        "cat_oof_ap": result["oof_default"]["average_precision"],
        "previous_cat_oof_auc": float(roc_auc_score(y, old_p)),
        "delta_vs_previous_cat": result["oof_default"]["roc_auc"] - float(roc_auc_score(y, old_p)),
        "fold_auc": [fold["metrics"]["roc_auc"] for fold in result["folds"]],
        "iterations": [fold["selected_iterations"] for fold in result["folds"]],
        "seconds": result["total_seconds"],
        "spearman_vs_previous_cat": float(pd.Series(new_p).corr(pd.Series(old_p), method="spearman")),
        "spearman_vs_xgb": float(pd.Series(new_p).corr(pd.Series(xgb_p), method="spearman")),
        "rows": rows,
        "best": best,
        "reference_blend_auc": float(roc_auc_score(y, reference)),
        "reference_blend_folds": reference_folds,
        "delta_vs_blend": best["oof_auc"] - REFERENCE_BLEND,
        "fold_deltas_vs_blend": fold_deltas,
        "spearman_vs_blend": float(pd.Series(best_p).corr(pd.Series(reference), method="spearman")),
        "improved": best["oof_auc"] >= REFERENCE_BLEND + 0.001 and float(pd.Series(best_p).corr(pd.Series(reference), method="spearman")) < 0.98 and all(delta > 0 for delta in fold_deltas),
    }
    write_json(ROOT / "reports/cat_glm.json", summary)
    print(json.dumps({
        "cat_oof_auc": summary["cat_oof_auc"],
        "delta_vs_previous_cat": summary["delta_vs_previous_cat"],
        "best_blend_auc": best["oof_auc"],
        "best_xgb_weight": best["xgb_weight"],
        "delta_vs_blend": summary["delta_vs_blend"],
        "fold_deltas_vs_blend": fold_deltas,
        "spearman_vs_blend": summary["spearman_vs_blend"],
        "spearman_vs_previous_cat": summary["spearman_vs_previous_cat"],
        "iterations": summary["iterations"],
        "improved": summary["improved"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
