"""用已选定的取值编码，在全部有标签数据上重训并生成比赛提交。不计算独立测试集指标，不上传。"""
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

EXPERIMENT = "xgb_r2_parts_smooth80"


def main():
    result = json.loads((ROOT / "reports/experiments" / f"{EXPERIMENT}.json").read_text(encoding="utf-8"))
    spec = result["spec"]
    rounds = [fold["selected_iterations"] for fold in result["folds"]]
    fixed = int(np.median(rounds))
    train, test = load_data()
    features = [column for column in train.columns if column not in {ID, TARGET}]
    bundle = fit_model(spec["kind"], spec["parameters"], train[features], train[TARGET], config()["seed"], spec["features"], fixed_iterations=fixed)
    probability = predict(bundle, test[features])
    submission = pd.DataFrame({ID: test[ID], TARGET: probability})
    sample = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    if list(submission.columns) != list(sample.columns) or not np.array_equal(submission[ID], sample[ID]):
        raise ValueError("提交列或 id 顺序不符合官方样例")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("提交概率无效")
    output = ROOT / "submissions" / f"{EXPERIMENT}.csv"
    output.parent.mkdir(exist_ok=True)
    submission.to_csv(output, index=False)
    model_path = ROOT / "artifacts" / "round2" / f"{EXPERIMENT}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)
    write_json(ROOT / "reports/value_encoding_submission.json", {
        "experiment": EXPERIMENT,
        "fixed_iterations": fixed,
        "fold_iterations": rounds,
        "rows": len(submission),
        "file": str(output.relative_to(ROOT)).replace("\\", "/"),
        "oof_auc": result["oof_default"]["roc_auc"],
        "trained_on": "all_official_labeled_data",
        "holdout_used_for_selection": False,
        "uploaded": False,
        "created_at": timestamp(),
    })
    print(f"已生成 {len(submission):,} 行，轮数 {fixed}，未上传：{output}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
