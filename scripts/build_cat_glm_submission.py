"""用全部有标签数据重训 GLM CatBoost，与已保存的深度 3 按 1:1 生成提交。"""
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
CAT_NAME = "cat_r3_glm_parts"
WEIGHT = 0.5


def main() -> None:
    blend = json.loads((ROOT / "reports/cat_glm.json").read_text(encoding="utf-8"))
    if abs(blend["best"]["xgb_weight"] - WEIGHT) > 1e-9:
        raise ValueError("融合权重与开发集选定结果不一致")
    cat_result = json.loads((ROOT / "reports/experiments" / f"{CAT_NAME}.json").read_text(encoding="utf-8"))
    rounds = [fold["selected_iterations"] for fold in cat_result["folds"]]
    fixed = int(np.median(rounds))
    train, test = load_data()
    features = [column for column in train.columns if column not in {ID, TARGET}]
    spec = cat_result["spec"]
    cat_bundle = fit_model(spec["kind"], spec["parameters"], train[features], train[TARGET], config()["seed"], spec["features"], fixed_iterations=fixed)
    if cat_bundle["training_device"] != "GPU":
        raise RuntimeError(f"CatBoost 全量重训未使用 GPU: {cat_bundle['training_device']}")
    xgb_bundle = joblib.load(ROOT / "artifacts/round2" / f"{XGB_NAME}.joblib")
    if xgb_bundle["selected_iterations"] != 410 or xgb_bundle["parameters"].get("value_encoding") != "parts":
        raise RuntimeError("已保存的深度 3 不是公开榜上的整数档小数档模型")
    if not str(xgb_bundle["training_device"]).startswith("cuda"):
        raise RuntimeError(f"已保存的深度 3 不是 CUDA 训练结果: {xgb_bundle['training_device']}")
    probability = WEIGHT * predict(xgb_bundle, test[features]) + (1 - WEIGHT) * predict(cat_bundle, test[features])
    submission = pd.DataFrame({ID: test[ID], TARGET: probability})
    sample = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    if list(submission.columns) != list(sample.columns) or not np.array_equal(submission[ID], sample[ID]):
        raise ValueError("提交列或 id 顺序不符合官方样例")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("提交概率无效")
    output = ROOT / "submissions/xgb_cat_glm_50_50.csv"
    output.parent.mkdir(exist_ok=True)
    submission.to_csv(output, index=False)
    folder = ROOT / "artifacts/round3"
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(cat_bundle, folder / f"{CAT_NAME}.joblib", compress=3)
    write_json(ROOT / "reports/cat_glm_submission.json", {
        "xgb": XGB_NAME,
        "cat": CAT_NAME,
        "xgb_weight": WEIGHT,
        "cat_weight": 1 - WEIGHT,
        "xgb_fixed_iterations": 410,
        "cat_fixed_iterations": fixed,
        "cat_fold_iterations": rounds,
        "oof_auc": blend["best"]["oof_auc"],
        "selection": "开发集折外预测上，深度 3 与 GLM CatBoost 按 0.1 步长搜索，0.5/0.5 最高。独立测试集不参与选型。",
        "rows": len(submission),
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
        "file": "submissions/xgb_cat_glm_50_50.csv",
        "trained_on": "all_official_labeled_data",
        "holdout_used_for_selection": False,
        "uploaded": False,
        "created_at": timestamp(),
    })
    print(f"已生成 {len(submission):,} 行，CatBoost 轮数 {fixed}：{output}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
