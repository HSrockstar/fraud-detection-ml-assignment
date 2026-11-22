"""在现有整数档和小数档上，只改变树的容量。不读取独立测试集。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud.common import ROOT, write_json
from fraud.experiments import run_cv

REFERENCE = "xgb_r2_parts_smooth80"
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
    "target_smooth": 80.0,
    "value_encoding": "parts",
}


def leaf_params(max_leaves: int) -> dict:
    params = dict(BASE)
    params.update(grow_policy="lossguide", max_leaves=max_leaves, max_depth=0)
    return params


def depth_params(max_depth: int) -> dict:
    params = dict(BASE)
    params["max_depth"] = max_depth
    return params


def booster_tree_param(name: str) -> dict:
    bundle = joblib.load(ROOT / "artifacts" / name / "fold_0.joblib")
    config = json.loads(bundle["model"].get_booster().save_config())
    train = config["learner"]["gradient_booster"]["tree_train_param"]
    return {
        "grow_policy": train["grow_policy"],
        "max_leaves": int(train["max_leaves"]) if "max_leaves" in train else None,
        "max_depth": int(train["max_depth"]),
        "device": config["learner"]["generic_param"]["device"],
    }


def main() -> None:
    specs = [
        ("xgb_r3_depth1_parts_s80", depth_params(1), {"grow_policy": "depthwise", "max_depth": 1}),
        ("xgb_r3_depth2_parts_s80", depth_params(2), {"grow_policy": "depthwise", "max_depth": 2}),
        ("xgb_r3_leaves2_parts_s80", leaf_params(2), {"grow_policy": "lossguide", "max_leaves": 2, "max_depth": 0}),
        ("xgb_r3_leaves3_parts_s80", leaf_params(3), {"grow_policy": "lossguide", "max_leaves": 3, "max_depth": 0}),
        ("xgb_r3_leaves4_parts_s80", leaf_params(4), {"grow_policy": "lossguide", "max_leaves": 4, "max_depth": 0}),
    ]
    reference = json.loads((ROOT / "reports/experiments" / f"{REFERENCE}.json").read_text(encoding="utf-8"))
    reference_auc = reference["oof_default"]["roc_auc"]
    rows = []
    for name, params, expected in specs:
        result = run_cv(name, "xgb", params, after_freeze=True)
        observed = booster_tree_param(name)
        devices = {fold["training_device"] for fold in result["folds"]}
        if not devices or any(not device.startswith("cuda") for device in devices):
            raise RuntimeError(f"{name} 未在 CUDA 上训练: {devices}")
        for key, value in expected.items():
            if observed[key] != value:
                raise RuntimeError(f"{name} 的 {key} 为 {observed[key]}，期望 {value}")
        auc = result["oof_default"]["roc_auc"]
        rows.append({
            "name": name,
            "oof_auc": auc,
            "oof_ap": result["oof_default"]["average_precision"],
            "fold_auc": [fold["metrics"]["roc_auc"] for fold in result["folds"]],
            "iterations": [fold["selected_iterations"] for fold in result["folds"]],
            "seconds": result["total_seconds"],
            "delta_vs_depth3": auc - reference_auc,
            "tree": observed,
        })
    summary = {
        "reference": REFERENCE,
        "reference_oof_auc": reference_auc,
        "kept_fixed": "整数档、小数档、目标编码平滑 80，以及深度 3 模型的学习率、采样和正则。只改深度或叶数。",
        "protocol": "开发集三折。早停只用外层训练折内部 10%。独立测试集不参与。",
        "rows": rows,
        "best": max(rows, key=lambda row: row["oof_auc"]),
    }
    write_json(ROOT / "reports/leaf_capacity.json", summary)
    print(json.dumps({"reference_oof_auc": reference_auc, "best": summary["best"]["name"], "best_auc": summary["best"]["oof_auc"], "delta": summary["best"]["delta_vs_depth3"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
