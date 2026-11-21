"""用全部有标签数据重训档位 CatBoost，与已保存的 XGBoost 按 7:3 生成提交。不计算独立测试集指标。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud.common import ID, ROOT, TARGET, config, load_data, timestamp, write_json
from fraud.models import fit_model, predict

XGB_NAME = "xgb_r2_parts_smooth80"
CAT_NAME = "cat_r2_parts"
XGB_WEIGHT = 0.7


def main():
    blend = json.loads((ROOT / "reports/cat_parts_blend.json").read_text(encoding="utf-8"))
    if abs(blend["best"]["xgb_weight"] - XGB_WEIGHT) > 1e-9:
        raise ValueError("融合权重与开发集选定结果不一致")
    cat_result = json.loads((ROOT / "reports/experiments" / f"{CAT_NAME}.json").read_text(encoding="utf-8"))
    rounds = [fold["selected_iterations"] for fold in cat_result["folds"]]
    fixed = int(np.median(rounds))
    train, test = load_data()
    features = [column for column in train.columns if column not in {ID, TARGET}]
    spec = cat_result["spec"]
    cat_bundle = fit_model(spec["kind"], spec["parameters"], train[features], train[TARGET], config()["seed"], spec["features"], fixed_iterations=fixed)
    xgb_bundle = joblib.load(ROOT / "artifacts/round2" / f"{XGB_NAME}.joblib")
    probability = XGB_WEIGHT * predict(xgb_bundle, test[features]) + (1 - XGB_WEIGHT) * predict(cat_bundle, test[features])
    submission = pd.DataFrame({ID: test[ID], TARGET: probability})
    sample = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    if list(submission.columns) != list(sample.columns) or not np.array_equal(submission[ID], sample[ID]):
        raise ValueError("提交列或 id 顺序不符合官方样例")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("提交概率无效")
    output = ROOT / "submissions/xgb_cat_parts_70_30.csv"
    output.parent.mkdir(exist_ok=True)
    submission.to_csv(output, index=False)
    model_path = ROOT / "artifacts/round2" / f"{CAT_NAME}.joblib"
    joblib.dump(cat_bundle, model_path, compress=3)
    write_json(ROOT / "reports/blend_submission.json", {
        "xgb": XGB_NAME,
        "cat": CAT_NAME,
        "xgb_weight": XGB_WEIGHT,
        "cat_weight": 1 - XGB_WEIGHT,
        "cat_fixed_iterations": fixed,
        "cat_fold_iterations": rounds,
        "oof_auc": blend["best"]["oof_auc"],
        "rows": len(submission),
        "file": "submissions/xgb_cat_parts_70_30.csv",
        "trained_on": "all_official_labeled_data",
        "holdout_used_for_selection": False,
        "uploaded": False,
        "created_at": timestamp(),
    })
    print(f"已生成 {len(submission):,} 行，CatBoost 轮数 {fixed}：{output}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
