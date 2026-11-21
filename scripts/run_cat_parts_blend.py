"""CatBoost 使用与当前 XGBoost 相同的整数档和小数档，再在开发集上搜索融合权重。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud.common import ID, ROOT, TARGET, config, write_json
from fraud.experiments import run_cv

XGB_NAME = "xgb_r2_parts_smooth80"
CAT_NAME = "cat_r2_parts"
MIN_GAIN = 0.00005


def blend_rows():
    cfg = config()["cat_baseline"]
    params = {**cfg, "value_encoding": "parts"}
    result = run_cv(CAT_NAME, "cat", params, after_freeze=True)
    xgb = pd.read_parquet(ROOT / "artifacts" / XGB_NAME / "oof.parquet")
    cat = pd.read_parquet(ROOT / "artifacts" / CAT_NAME / "oof.parquet")
    merged = xgb.merge(cat, on=ID, suffixes=("_xgb", "_cat"))
    if not np.array_equal(merged[f"{TARGET}_xgb"], merged[f"{TARGET}_cat"]):
        raise ValueError("两份折外预测的标签不一致")
    if not np.array_equal(merged["fold_xgb"], merged["fold_cat"]):
        raise ValueError("两份折外预测的折号不一致")
    y = merged[f"{TARGET}_xgb"].to_numpy()
    xgb_p = merged["probability_xgb"].to_numpy()
    cat_p = merged["probability_cat"].to_numpy()
    rows = []
    for step in range(11):
        weight = step / 10
        probability = weight * xgb_p + (1 - weight) * cat_p
        rows.append({
            "xgb_weight": weight,
            "cat_weight": 1 - weight,
            "oof_auc": float(roc_auc_score(y, probability)),
            "oof_ap": float(average_precision_score(y, probability)),
        })
    best_single = max(rows[0]["oof_auc"], rows[-1]["oof_auc"])
    best = max(rows, key=lambda row: row["oof_auc"])
    summary = {
        "selection": "开发集 OOF ROC-AUC。融合需比最好的单模型至少高 0.00005。独立测试集不参与。",
        "xgb": XGB_NAME,
        "cat": CAT_NAME,
        "cat_oof_auc": result["oof_default"]["roc_auc"],
        "cat_oof_ap": result["oof_default"]["average_precision"],
        "cat_seconds": result["total_seconds"],
        "rows": rows,
        "best": best,
        "best_single_auc": best_single,
        "gain_over_best_single": best["oof_auc"] - best_single,
        "adopt": best["oof_auc"] >= best_single + MIN_GAIN and best["cat_weight"] not in {0.0, 1.0},
    }
    write_json(ROOT / "reports/cat_parts_blend.json", summary)
    print(json.dumps({"cat_oof_auc": summary["cat_oof_auc"], "best": best, "adopt": summary["adopt"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    blend_rows()
