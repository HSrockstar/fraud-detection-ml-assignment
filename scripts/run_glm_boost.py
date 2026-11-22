"""深度 3 的整数档和小数档上，用逻辑回归 logit 作为 XGBoost 的起点。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud.common import ID, ROOT, TARGET, write_json
from fraud.experiments import run_cv

REFERENCE_BLEND = 0.9478163587782216
REFERENCE_XGB = "xgb_r2_parts_smooth80"
UPLOAD_GAIN = 0.001
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
    "target_smooth": 80.0,
    "value_encoding": "parts",
    "glm_boost": True,
    "glm_C": 0.01,
}


def assert_margin_reaches_prediction() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 4)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int32)
    margin = np.full(len(x), 4.0, dtype=np.float32)
    model = XGBClassifier(
        n_estimators=1, max_depth=1, learning_rate=0.01, tree_method="hist", device="cuda",
        objective="binary:logistic", eval_metric="auc",
    )
    model.fit(x, y, base_margin=margin, verbose=False)
    device = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
    if not device.startswith("cuda"):
        raise RuntimeError("GLM 起点检查未在 CUDA 上训练")
    model.set_params(device="cpu")
    with_margin = model.predict_proba(x, base_margin=margin)[:, 1]
    without_margin = model.predict_proba(x, base_margin=np.zeros(len(x), dtype=np.float32))[:, 1]
    if not np.isfinite(with_margin).all() or with_margin.mean() <= without_margin.mean():
        raise RuntimeError("base_margin 没有进入预测")


def blend_with_cat(glm_oof: pd.DataFrame) -> dict:
    cat = pd.read_parquet(ROOT / "artifacts/cat_r2_parts/oof.parquet")
    merged = glm_oof.merge(cat, on=ID, suffixes=("_glm", "_cat"))
    y = merged[f"{TARGET}_glm"].to_numpy()
    if not np.array_equal(y, merged[f"{TARGET}_cat"].to_numpy()):
        raise ValueError("CatBoost 折外标签与 GLM 树不一致")
    glm_p = merged["probability_glm"].to_numpy()
    cat_p = merged["probability_cat"].to_numpy()
    rows = []
    for step in range(11):
        weight = step / 10
        rows.append({"glm_weight": weight, "cat_weight": 1 - weight, "oof_auc": float(roc_auc_score(y, weight * glm_p + (1 - weight) * cat_p))})
    best = max(rows, key=lambda row: row["oof_auc"])
    return {"rows": rows, "best": best}


def main() -> None:
    assert_margin_reaches_prediction()
    result = run_cv("xgb_r3_glm_parts_s80", "xgb", PARAMS, after_freeze=True)
    devices = {fold["training_device"] for fold in result["folds"]}
    if any(not device.startswith("cuda") for device in devices):
        raise RuntimeError(f"未在 CUDA 上训练: {devices}")
    reference = json.loads((ROOT / "reports/experiments" / f"{REFERENCE_XGB}.json").read_text(encoding="utf-8"))
    auc = result["oof_default"]["roc_auc"]
    oof = pd.read_parquet(ROOT / "artifacts/xgb_r3_glm_parts_s80/oof.parquet")
    blend = blend_with_cat(oof)
    summary = {
        "name": "xgb_r3_glm_parts_s80",
        "glm_C": 0.01,
        "base_model": REFERENCE_XGB,
        "oof_auc": auc,
        "oof_ap": result["oof_default"]["average_precision"],
        "fold_auc": [fold["metrics"]["roc_auc"] for fold in result["folds"]],
        "iterations": [fold["selected_iterations"] for fold in result["folds"]],
        "seconds": result["total_seconds"],
        "depth3_oof_auc": reference["oof_default"]["roc_auc"],
        "delta_vs_depth3": auc - reference["oof_default"]["roc_auc"],
        "blend_7_3_oof_auc": REFERENCE_BLEND,
        "delta_vs_blend": auc - REFERENCE_BLEND,
        "cat_blend": blend["best"],
        "upload_rule": "开发集 OOF 需比当前公开有效的 7:3 融合至少高 0.001。浅树曾高 0.0005 但公开榜下降。",
        "upload": max(auc, blend["best"]["oof_auc"]) >= REFERENCE_BLEND + UPLOAD_GAIN,
    }
    write_json(ROOT / "reports/glm_boost.json", summary)
    print(json.dumps({
        "oof_auc": auc,
        "delta_vs_depth3": summary["delta_vs_depth3"],
        "delta_vs_blend": summary["delta_vs_blend"],
        "cat_blend": blend["best"],
        "upload": summary["upload"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
