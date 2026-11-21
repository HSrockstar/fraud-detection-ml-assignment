from __future__ import annotations

import gc
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .common import ID, ROOT, TARGET, config, load_data, splits, timestamp, write_json
from .evaluation import f1_threshold, metrics
from .models import fit_model, predict


def select():
    path = ROOT / "reports/selection.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    results = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "reports/experiments").glob("*.json"))}
    if not {"logistic_baseline", "tree_baseline", "xgb_baseline", "cat_baseline"}.issubset(results):
        raise RuntimeError("请先完成所有基线")
    oofs = {name: pd.read_parquet(ROOT / "artifacts" / name / "oof.parquet") for name in results}
    reference = next(iter(oofs.values()))
    if any(not value[[ID, TARGET, "fold"]].equals(reference[[ID, TARGET, "fold"]]) for value in oofs.values()):
        raise ValueError("OOF 样本或折划分不一致，禁止融合")
    y = reference[TARGET].to_numpy()
    candidates = [{"weights": {name: 1.0}, "auc": float(roc_auc_score(y, value.probability))} for name, value in oofs.items()]
    single = max(candidates, key=lambda c: c["auc"])
    if "cat_baseline" in results:
        for name in results:
            if results[name]["spec"]["kind"] != "xgb":
                continue
            for weight in np.linspace(0.1, 0.9, 9):
                weights = {name: float(weight), "cat_baseline": float(1 - weight)}
                p = sum(oofs[n].probability.to_numpy() * w for n, w in weights.items())
                candidates.append({"weights": weights, "auc": float(roc_auc_score(y, p))})
    best = max(candidates, key=lambda c: c["auc"])
    # 极小且未经独立确认的提升不增加推理成本。
    if best["auc"] - single["auc"] < 0.00005:
        best = single
    probabilities = sum(oofs[n].probability.to_numpy() * w for n, w in best["weights"].items())
    threshold = f1_threshold(y, probabilities)
    result = {"frozen_at": timestamp(), "criterion": "开发集 OOF ROC-AUC；融合比最佳单模型至少提升 0.00005 才采用",
              "weights": best["weights"], "oof_metrics": metrics(y, probabilities, threshold),
              "best_single": single, "candidates": candidates, "f1_threshold": threshold,
              "selection_note": "OOF 用于选参、选权重和选阈值，其选后成绩可能乐观；独立测试成绩另列。",
              "all_experiments": list(results)}
    # 此文件在任何 holdout 预测之前持久化，后续不依据 holdout 成绩更改。
    write_json(path, result)
    pd.DataFrame({ID: reference[ID], TARGET: y, "probability": probabilities}).to_parquet(ROOT / "artifacts/selected_oof.parquet", index=False)
    return result


def run():
    complete = ROOT / "reports/finalization.json"
    if complete.exists():
        print("本轮最终评估和提交文件已完成，保留原始测试结果。", flush=True)
        return
    selection = select()
    cfg = config()
    train, test = load_data()
    partition = splits(train).partition
    dev, holdout = train.loc[partition == "dev"], train.loc[partition == "holdout"]
    feature_columns = train.columns.drop([ID, TARGET]).tolist()
    results = {name: json.loads((ROOT / "reports/experiments" / f"{name}.json").read_text(encoding="utf-8")) for name in selection["all_experiments"]}
    evaluation_folder = ROOT / "artifacts/evaluation"
    evaluation_folder.mkdir(parents=True, exist_ok=True)
    holdout_probs, holdout_results = {}, {}
    rounds = {}
    for name, result in results.items():
        spec = result["spec"]
        values = [f["selected_iterations"] for f in result["folds"] if f["selected_iterations"] is not None]
        rounds[name] = int(np.median(values)) if values else None
        model_path = evaluation_folder / f"{name}.joblib"
        if model_path.exists():
            bundle = joblib.load(model_path)
        else:
            bundle = fit_model(spec["kind"], spec["parameters"], dev[feature_columns], dev[TARGET], cfg["seed"], spec["features"], fixed_iterations=rounds[name])
            joblib.dump(bundle, model_path, compress=3)
        probabilities = predict(bundle, holdout[feature_columns])
        threshold = result["oof_selected_threshold"]["threshold"]
        holdout_probs[name] = probabilities
        holdout_results[name] = {"default_threshold": metrics(holdout[TARGET], probabilities), "oof_threshold": metrics(holdout[TARGET], probabilities, threshold),
                                 "training_seconds": bundle["training_seconds"], "fixed_iterations": rounds[name], "training_device": bundle["training_device"]}
        print(f"最终测试 {name}: AUC={holdout_results[name]['oof_threshold']['roc_auc']:.6f}，选型保持冻结", flush=True)
        del bundle
        gc.collect()
    selected_prob = sum(holdout_probs[n] * w for n, w in selection["weights"].items())
    holdout_results["selected"] = {"default_threshold": metrics(holdout[TARGET], selected_prob),
                                    "oof_threshold": metrics(holdout[TARGET], selected_prob, selection["f1_threshold"])}
    write_json(ROOT / "reports/holdout_metrics.json", holdout_results)
    predictions = pd.DataFrame({ID: holdout[ID].to_numpy(), TARGET: holdout[TARGET].to_numpy(), **holdout_probs, "selected": selected_prob})
    predictions.to_parquet(evaluation_folder / "holdout_predictions.parquet", index=False)
    # 完成独立评估后，用全部有标签数据重训比赛模型。
    submission_folder = ROOT / "submissions"
    submission_folder.mkdir(exist_ok=True)
    final_folder = ROOT / "artifacts/final"
    final_folder.mkdir(parents=True, exist_ok=True)
    test_probs = {}
    for name in selection["weights"]:
        spec = results[name]["spec"]
        bundle = fit_model(spec["kind"], spec["parameters"], train[feature_columns], train[TARGET], cfg["seed"], spec["features"], fixed_iterations=rounds[name])
        test_probs[name] = predict(bundle, test[feature_columns])
        joblib.dump(bundle, final_folder / f"{name}.joblib", compress=3)
        pd.DataFrame({ID: test[ID], TARGET: test_probs[name]}).to_csv(submission_folder / f"round1_{name}.csv", index=False)
        del bundle
        gc.collect()
    final_probability = sum(test_probs[n] * w for n, w in selection["weights"].items())
    submission = pd.DataFrame({ID: test[ID], TARGET: final_probability})
    sample = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    if list(submission.columns) != list(sample.columns) or not np.array_equal(submission[ID], sample[ID]):
        raise ValueError("提交列或 id 顺序不符合官方样例")
    if not np.isfinite(final_probability).all() or ((final_probability < 0) | (final_probability > 1)).any():
        raise ValueError("提交概率无效")
    submission.to_csv(submission_folder / "round1_best.csv", index=False)
    write_json(final_folder / "manifest.json", {"feature_columns": feature_columns, "weights": selection["weights"], "id_column": ID, "target_column": TARGET, "trained_on": "all_official_labeled_data"})
    write_json(complete, {"completed_at": timestamp(), "selection_frozen_at": selection["frozen_at"], "weights": selection["weights"],
                         "submission": "submissions/round1_best.csv", "rows": len(submission), "fixed_iterations": rounds,
                         "holdout_auc": holdout_results["selected"]["oof_threshold"]["roc_auc"],
                         "kaggle_submission_status": "仅生成文件，未上传 Kaggle；无线上分数。"})
