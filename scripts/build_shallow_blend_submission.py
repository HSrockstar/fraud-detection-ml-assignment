"""用全部有标签数据重训 3 叶和深度 1，按开发集选定的 1:1 生成提交。"""
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

LEAVES = "xgb_r3_leaves3_parts_s80"
DEPTH = "xgb_r3_depth1_parts_s80"
WEIGHT = 0.5


def median_rounds(name: str) -> tuple[int, list[int], dict]:
    result = json.loads((ROOT / "reports/experiments" / f"{name}.json").read_text(encoding="utf-8"))
    rounds = [fold["selected_iterations"] for fold in result["folds"]]
    return int(np.median(rounds)), rounds, result["spec"]


def main() -> None:
    leaves_rounds, leaves_fold_rounds, leaves_spec = median_rounds(LEAVES)
    depth_rounds, depth_fold_rounds, depth_spec = median_rounds(DEPTH)
    train, test = load_data()
    features = [column for column in train.columns if column not in {ID, TARGET}]
    seed = config()["seed"]
    leaves_bundle = fit_model(leaves_spec["kind"], leaves_spec["parameters"], train[features], train[TARGET], seed, leaves_spec["features"], fixed_iterations=leaves_rounds)
    depth_bundle = fit_model(depth_spec["kind"], depth_spec["parameters"], train[features], train[TARGET], seed, depth_spec["features"], fixed_iterations=depth_rounds)
    if not leaves_bundle["training_device"].startswith("cuda") or not depth_bundle["training_device"].startswith("cuda"):
        raise RuntimeError("全量重训未使用 CUDA")
    probability = WEIGHT * predict(leaves_bundle, test[features]) + (1 - WEIGHT) * predict(depth_bundle, test[features])
    submission = pd.DataFrame({ID: test[ID], TARGET: probability})
    sample = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    if list(submission.columns) != list(sample.columns) or not np.array_equal(submission[ID], sample[ID]):
        raise ValueError("提交列或 id 顺序不符合官方样例")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("提交概率无效")
    output = ROOT / "submissions/xgb_leaves3_depth1_50_50.csv"
    output.parent.mkdir(exist_ok=True)
    submission.to_csv(output, index=False)
    folder = ROOT / "artifacts/round3"
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(leaves_bundle, folder / f"{LEAVES}.joblib", compress=3)
    joblib.dump(depth_bundle, folder / f"{DEPTH}.joblib", compress=3)
    write_json(ROOT / "reports/shallow_blend_submission.json", {
        "leaves": LEAVES,
        "depth1": DEPTH,
        "leaves_weight": WEIGHT,
        "depth1_weight": 1 - WEIGHT,
        "leaves_fixed_iterations": leaves_rounds,
        "depth1_fixed_iterations": depth_rounds,
        "leaves_fold_iterations": leaves_fold_rounds,
        "depth1_fold_iterations": depth_fold_rounds,
        "oof_auc": 0.9483087565923245,
        "selection": "开发集折外预测上，3 叶与深度 1 按 0.05 步长搜索，0.5/0.5 最高。加入 CatBoost 的增益低于 0.00005，未纳入。独立测试集不参与。",
        "rows": len(submission),
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
        "file": "submissions/xgb_leaves3_depth1_50_50.csv",
        "trained_on": "all_official_labeled_data",
        "holdout_used_for_selection": False,
        "uploaded": False,
        "created_at": timestamp(),
    })
    print(f"已生成 {len(submission):,} 行，3 叶 {leaves_rounds} 轮，深度 1 {depth_rounds} 轮：{output}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
