from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

from .common import ID, ROOT, TARGET, config, load_data, splits, timestamp, write_json
from .evaluation import f1_threshold, metrics
from .models import feature_importance, fit_model, predict


def development_data():
    train, test = load_data()
    split = splits(train)
    mask = split.partition == "dev"
    dev = train.loc[mask].reset_index(drop=True)
    folds = split.loc[mask, "fold"].to_numpy()
    return dev, folds, test


def result_path(name):
    return ROOT / "reports/experiments" / f"{name}.json"


def assert_selection_open():
    if (ROOT / "reports/selection.json").exists():
        raise RuntimeError("本轮已冻结模型选择。后续改进仍可使用开发集，但须另建实验轮次并披露既有测试集已被评估。")


def run_cv(name, kind, params, features=False, after_freeze=False):
    cfg = config()
    spec = {"name": name, "kind": kind, "parameters": params, "features": features}
    if result_path(name).exists():
        existing = json.loads(result_path(name).read_text(encoding="utf-8"))
        if existing["spec"] != spec or existing["config"] != cfg:
            raise ValueError(f"实验 {name} 已存在且参数不同，请使用新的名称")
        print(f"复用已完成实验 {name}", flush=True)
        return existing
    if not after_freeze:
        assert_selection_open()
    dev, folds, _ = development_data()
    x, y = dev.drop(columns=[ID, TARGET]), dev[TARGET].to_numpy()
    folder = ROOT / "artifacts" / name
    folder.mkdir(parents=True, exist_ok=True)
    probabilities = np.full(len(dev), np.nan)
    fold_results = []
    started = time.perf_counter()
    for fold in range(cfg["n_splits"]):
        train_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        fold_file = folder / f"fold_{fold}.joblib"
        # 未完成的折重新训练；完整实验由 JSON 结果作为提交标志。
        bundle = fit_model(kind, params, x.iloc[train_idx], y[train_idx], cfg["seed"] + fold, features)
        probability = predict(bundle, x.iloc[valid_idx])
        probabilities[valid_idx] = probability
        fold_result = {"fold": fold, "train_rows": len(train_idx), "valid_rows": len(valid_idx),
                       "training_device": bundle["training_device"], "training_seconds": bundle["training_seconds"],
                       "selected_iterations": bundle["selected_iterations"], "metrics": metrics(y[valid_idx], probability)}
        fold_results.append(fold_result)
        joblib.dump(bundle, fold_file, compress=3)
        write_json(folder / f"fold_{fold}.json", fold_result)
        print(f"{name} fold={fold} AUC={fold_result['metrics']['roc_auc']:.6f} AP={fold_result['metrics']['average_precision']:.6f} seconds={bundle['training_seconds']:.1f} rounds={bundle['selected_iterations']}", flush=True)
        if fold == 0:
            write_json(folder / "feature_importance.json", feature_importance(bundle))
        del bundle
        gc.collect()
    if not np.isfinite(probabilities).all():
        raise RuntimeError("OOF 预测未覆盖所有开发样本")
    threshold = f1_threshold(y, probabilities)
    pd.DataFrame({ID: dev[ID], "fold": folds, TARGET: y, "probability": probabilities}).to_parquet(folder / "oof.parquet", index=False)
    result = {"spec": spec, "config": cfg, "completed_at": timestamp(),
              "source_sha256": {n: hashlib.sha256((ROOT / "src/fraud" / n).read_bytes()).hexdigest() for n in ["models.py", "features.py", "evaluation.py"]},
              "versions": {p: importlib.metadata.version(p) for p in ["numpy", "scikit-learn", "xgboost", "catboost"]},
              "folds": fold_results, "oof_default": metrics(y, probabilities), "oof_selected_threshold": metrics(y, probabilities, threshold),
              "mean_fold_auc": float(np.mean([f["metrics"]["roc_auc"] for f in fold_results])),
              "std_fold_auc": float(np.std([f["metrics"]["roc_auc"] for f in fold_results], ddof=1)),
              "total_seconds": time.perf_counter() - started}
    write_json(result_path(name), result)
    if not after_freeze:
        refresh_comparison()
    return result


def refresh_comparison():
    rows = []
    for path in sorted((ROOT / "reports/experiments").glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"name": result["spec"]["name"], "kind": result["spec"]["kind"], "feature_engineering": result["spec"]["features"],
                     **{k: v for k, v in result["oof_selected_threshold"].items() if k != "confusion_matrix"},
                     "mean_fold_auc": result["mean_fold_auc"], "std_fold_auc": result["std_fold_auc"], "seconds": result["total_seconds"]})
    pd.DataFrame(rows).sort_values("roc_auc", ascending=False).to_csv(ROOT / "reports/model_comparison.csv", index=False, encoding="utf-8-sig")


def baseline():
    cfg = config()
    for name, kind, params in [("logistic_baseline", "logistic", {}), ("tree_baseline", "tree", {}),
                               ("xgb_baseline", "xgb", cfg["xgb_baseline"]), ("cat_baseline", "cat", cfg["cat_baseline"])]:
        run_cv(name, kind, params)


def optimize(trials=None):
    assert_selection_open()
    cfg = config()
    trials = trials or cfg["optimization_trials"]
    dev, folds, _ = development_data()
    x, y = dev.drop(columns=[ID, TARGET]), dev[TARGET].to_numpy()
    train_idx, valid_idx = np.flatnonzero(folds != 0), np.flatnonzero(folds == 0)
    # 只在开发集第 0 折搜索，之后用完整 3 折确认；搜索分数不当作独立泛化成绩。
    storage = "sqlite:///" + (ROOT / "artifacts/optuna.sqlite3").as_posix()
    study = optuna.create_study(study_name="xgb_round1", storage=storage, load_if_exists=True, direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=cfg["seed"], n_startup_trials=5))
    def objective(trial):
        params = {"n_estimators": 2000, "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.09, log=True),
                  "max_depth": trial.suggest_int("max_depth", 3, 7), "min_child_weight": trial.suggest_float("min_child_weight", 5, 100, log=True),
                  "subsample": trial.suggest_float("subsample", 0.65, 1.0), "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                  "reg_lambda": trial.suggest_float("reg_lambda", 1, 100, log=True), "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 3, log=True), "max_bin": 256}
        bundle = fit_model("xgb", params, x.iloc[train_idx], y[train_idx], cfg["seed"], features=False)
        score = float(roc_auc_score(y[valid_idx], predict(bundle, x.iloc[valid_idx])))
        trial.set_user_attr("selected_iterations", bundle["selected_iterations"])
        trial.set_user_attr("training_seconds", bundle["training_seconds"])
        del bundle
        gc.collect()
        return score
    completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if completed < trials:
        study.optimize(objective, n_trials=trials - completed, n_jobs=1, gc_after_trial=True)
    study.trials_dataframe().to_csv(ROOT / "reports/optimization_trials.csv", index=False)
    best_params = {**study.best_params, "n_estimators": 2000, "max_bin": 256}
    write_json(ROOT / "reports/optimization.json", {"best_params": best_params, "search_fold": 0, "search_auc": study.best_value,
               "completed_trials": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
               "note": "该搜索折参与了参数选择，其分数有选择偏差；最终报告另列独立测试集评估。"})
    run_cv("xgb_tuned", "xgb", best_params)
    # 单独比较特征工程增益，避免把参数和特征改变混为一谈。
    run_cv("xgb_engineered", "xgb", best_params, features=True)
    run_cv("xgb_target_encoded", "xgb", {**best_params, "target_encoding": True})
