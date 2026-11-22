"""在 3 叶模型上增加两位尾数和十位分组，只和同一棵浅树比较。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud.common import ROOT, write_json
from fraud.experiments import run_cv

REFERENCE = "xgb_r3_leaves3_parts_s80"
PARAMS = {
    "learning_rate": 0.0886816342974295,
    "max_depth": 0,
    "min_child_weight": 18.02471215771192,
    "subsample": 0.9974913605156784,
    "colsample_bytree": 0.9996687449838664,
    "reg_lambda": 1.1500035567289912,
    "reg_alpha": 2.7305861513999963,
    "n_estimators": 2000,
    "max_bin": 256,
    "grow_policy": "lossguide",
    "max_leaves": 3,
    "target_encoding": True,
    "target_smooth": 80.0,
    "value_encoding": "parts+tails",
}


def main() -> None:
    reference = json.loads((ROOT / "reports/experiments" / f"{REFERENCE}.json").read_text(encoding="utf-8"))
    reference_auc = reference["oof_default"]["roc_auc"]
    result = run_cv("xgb_r3_leaves3_parts_tails_s80", "xgb", PARAMS, after_freeze=True)
    devices = {fold["training_device"] for fold in result["folds"]}
    if any(not device.startswith("cuda") for device in devices):
        raise RuntimeError(f"未在 CUDA 上训练: {devices}")
    auc = result["oof_default"]["roc_auc"]
    summary = {
        "reference": REFERENCE,
        "reference_oof_auc": reference_auc,
        "added": "六列数值的两位小数尾数，以及整数按 10 分组。原有整数档和四位小数档保留。",
        "name": "xgb_r3_leaves3_parts_tails_s80",
        "oof_auc": auc,
        "oof_ap": result["oof_default"]["average_precision"],
        "fold_auc": [fold["metrics"]["roc_auc"] for fold in result["folds"]],
        "iterations": [fold["selected_iterations"] for fold in result["folds"]],
        "seconds": result["total_seconds"],
        "delta_vs_leaves3": auc - reference_auc,
        "adopt": auc >= reference_auc + 0.00005,
    }
    write_json(ROOT / "reports/tail_keys.json", summary)
    print(json.dumps({"oof_auc": auc, "delta_vs_leaves3": summary["delta_vs_leaves3"], "adopt": summary["adopt"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
