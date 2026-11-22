"""用全部有标签数据重训深度 3 的 GLM 起点模型并生成提交。"""
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

NAME = "xgb_r3_glm_parts_s80"


def main() -> None:
    result = json.loads((ROOT / "reports/experiments" / f"{NAME}.json").read_text(encoding="utf-8"))
    rounds = [fold["selected_iterations"] for fold in result["folds"]]
    fixed = int(np.median(rounds))
    train, test = load_data()
    features = [column for column in train.columns if column not in {ID, TARGET}]
    spec = result["spec"]
    bundle = fit_model(spec["kind"], spec["parameters"], train[features], train[TARGET], config()["seed"], spec["features"], fixed_iterations=fixed)
    if not str(bundle["training_device"]).startswith("cuda"):
        raise RuntimeError(f"全量重训未使用 CUDA: {bundle['training_device']}")
    if bundle.get("glm") is None:
        raise RuntimeError("全量模型没有逻辑回归起点")
    probability = predict(bundle, test[features])
    submission = pd.DataFrame({ID: test[ID], TARGET: probability})
    sample = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    if list(submission.columns) != list(sample.columns) or not np.array_equal(submission[ID], sample[ID]):
        raise ValueError("提交列或 id 顺序不符合官方样例")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("提交概率无效")
    output = ROOT / "submissions" / f"{NAME}.csv"
    output.parent.mkdir(exist_ok=True)
    submission.to_csv(output, index=False)
    folder = ROOT / "artifacts/round3"
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, folder / f"{NAME}.joblib", compress=3)
    record = json.loads((ROOT / "reports/glm_boost.json").read_text(encoding="utf-8"))
    write_json(ROOT / "reports/glm_submission.json", {
        "experiment": NAME,
        "fixed_iterations": fixed,
        "fold_iterations": rounds,
        "oof_auc": record["oof_auc"],
        "delta_vs_blend": record["delta_vs_blend"],
        "rows": len(submission),
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
        "file": f"submissions/{NAME}.csv",
        "trained_on": "all_official_labeled_data",
        "holdout_used_for_selection": False,
        "uploaded": False,
        "created_at": timestamp(),
    })
    print(f"已生成 {len(submission):,} 行，轮数 {fixed}：{output}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
