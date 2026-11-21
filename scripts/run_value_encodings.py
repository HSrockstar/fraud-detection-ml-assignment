"""第二轮取值编码。开发集折外 AUC 决定取舍，不读取独立测试集，也不改冻结选型。"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud.common import ID, ROOT, TARGET, config, write_json
from fraud.experiments import development_data, run_cv
from fraud.models import fit_model, predict

BASE = {
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
}
CANDIDATES = (
    ("exact", "exact"),
    ("parts", "parts"),
    ("digits", "digits"),
    ("interactions", "interactions"),
    ("interactions_ext", "interactions+interactions_ext"),
    ("exact_digits", "exact+digits"),
    ("full", "exact+parts+digits+interactions+interactions_ext"),
)
PUBLISHED_TARGET_ENCODED_OOF_AUC = 0.9456986304604758
PUBLISHED_BLEND_OOF_AUC = 0.9462735237853402


def params_for(mode=None, smooth=20.0):
    params = dict(BASE)
    params["target_smooth"] = smooth
    if mode:
        params["value_encoding"] = mode
    return params


def fold_auc(params):
    cfg = config()
    dev, folds, _ = development_data()
    x, y = dev.drop(columns=[ID, TARGET]), dev[TARGET].to_numpy()
    train_idx, valid_idx = np.flatnonzero(folds != 0), np.flatnonzero(folds == 0)
    bundle = fit_model("xgb", params, x.iloc[train_idx], y[train_idx], cfg["seed"], features=False)
    score = float(roc_auc_score(y[valid_idx], predict(bundle, x.iloc[valid_idx])))
    seconds = bundle["training_seconds"]
    del bundle
    gc.collect()
    return score, seconds


def summarize(name, mode, result):
    return {
        "name": name,
        "value_encoding": mode,
        "target_smooth": result["spec"]["parameters"].get("target_smooth", 20.0),
        "oof_auc": result["oof_default"]["roc_auc"],
        "oof_ap": result["oof_default"]["average_precision"],
        "std_fold_auc": result["std_fold_auc"],
        "seconds": result["total_seconds"],
    }


def main():
    screen = []
    reference_fold, reference_seconds = fold_auc(params_for())
    screen.append({"name": "reference", "value_encoding": None, "fold0_auc": reference_fold, "seconds": reference_seconds})
    print(f"筛选 reference fold0 AUC={reference_fold:.6f} seconds={reference_seconds:.1f}", flush=True)
    for name, mode in CANDIDATES:
        score, seconds = fold_auc(params_for(mode))
        screen.append({"name": name, "value_encoding": mode, "fold0_auc": score, "seconds": seconds})
        print(f"筛选 {name} fold0 AUC={score:.6f} delta={score - reference_fold:+.6f} seconds={seconds:.1f}", flush=True)

    ranked = sorted((row for row in screen if row["value_encoding"]), key=lambda row: row["fold0_auc"], reverse=True)
    chosen = []
    for row in ranked:
        if row["fold0_auc"] >= reference_fold - 0.0002 or row["name"] in {"exact_digits", "full"}:
            chosen.append(row)
        if len(chosen) == 3:
            break
    if not any(row["name"] == "exact_digits" for row in chosen):
        chosen.append(next(row for row in ranked if row["name"] == "exact_digits"))

    rows = [summarize("xgb_r2_reference", None, run_cv("xgb_r2_reference", "xgb", params_for(), after_freeze=True))]
    print(f"三折 xgb_r2_reference AUC={rows[0]['oof_auc']:.6f}", flush=True)
    for row in chosen:
        name = f"xgb_r2_{row['name']}"
        result = run_cv(name, "xgb", params_for(row["value_encoding"]), after_freeze=True)
        rows.append(summarize(name, row["value_encoding"], result))
        print(f"三折 {name} AUC={rows[-1]['oof_auc']:.6f}", flush=True)

    reference = rows[0]["oof_auc"]
    best = max(rows, key=lambda row: row["oof_auc"])
    if best["value_encoding"] and best["oof_auc"] > reference:
        for smooth, suffix in ((5.0, "smooth5"), (80.0, "smooth80")):
            name = f"{best['name']}_{suffix}"
            result = run_cv(name, "xgb", params_for(best["value_encoding"], smooth), after_freeze=True)
            rows.append(summarize(name, best["value_encoding"], result))
            print(f"三折 {name} AUC={rows[-1]['oof_auc']:.6f}", flush=True)
        best = max(rows, key=lambda row: row["oof_auc"])

    summary = {
        "selection": "开发集 OOF ROC-AUC。第 0 折只用于筛选，三折结果才用于取舍。独立测试集不参与。",
        "published_target_encoded_oof_auc": PUBLISHED_TARGET_ENCODED_OOF_AUC,
        "published_blend_oof_auc": PUBLISHED_BLEND_OOF_AUC,
        "fold0_screen": screen,
        "rows": rows,
        "best": best,
    }
    write_json(ROOT / "reports/value_encoding_round.json", summary)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
