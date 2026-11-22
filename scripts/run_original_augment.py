"""把 CC0 原始样本加入训练折，只在开发集折外预测上和当前方案比较。

原始行只要和比赛训练或测试的特征整行相同就剔除，避免把验证折或比赛测试的标签漏进训练。
整行重复达到原始样本的一半时不训练。独立测试集不参与比较，也不生成新的提交文件，除非折外融合明显高于当前 0.947816。
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud.common import ID, ROOT, TARGET, config, load_data, splits, timestamp, write_json
from fraud.evaluation import f1_threshold, metrics
from fraud.models import feature_importance, fit_model, predict

DATASET = "uditjain13/credit-card-fraud-detection-2026"
ORIGINAL_PATH = ROOT / "data/original/credit_card_fraud_2026.csv"
REFERENCE_EXPERIMENT = "xgb_r2_parts_smooth80"
CAT_OOF = ROOT / "artifacts/cat_r2_parts/oof.parquet"
NAME = "xgb_r3_original_parts"
CURRENT_BLEND_OOF_AUC = 0.9478163587782216
DUPLICATE_STOP = 0.5
CLEAR_GAIN = 0.001
VALUE_COLUMNS = (
    "amount_usd", "account_balance_usd", "distance_from_home_km",
    "hours_since_last_txn", "velocity_score", "merchant_risk_score",
)


def feature_columns(train: pd.DataFrame) -> list[str]:
    return [column for column in train.columns if column not in {ID, TARGET}]


def load_original(columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    if not ORIGINAL_PATH.exists():
        ORIGINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(DATASET, path=str(ORIGINAL_PATH.parent), unzip=True, quiet=False)
    original = pd.read_csv(ORIGINAL_PATH)
    missing = [column for column in columns + [TARGET] if column not in original.columns]
    if missing:
        raise ValueError(f"原始数据缺少列: {missing}")
    if not set(original[TARGET].dropna().unique()) <= {0, 1}:
        raise ValueError("原始标签不是 0/1")
    extra = [column for column in original.columns if column not in columns + [TARGET]]
    return original[columns + [TARGET]].copy(), extra


def row_hashes(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    normalized = pd.DataFrame(index=frame.index)
    for column in columns:
        series = frame[column]
        if pd.api.types.is_bool_dtype(series):
            normalized[column] = series.astype("int8")
        elif pd.api.types.is_float_dtype(series):
            normalized[column] = np.round(series.to_numpy(dtype="float64"), 8)
        else:
            normalized[column] = series.astype("string")
    return pd.util.hash_pandas_object(normalized, index=False).to_numpy()


def column_overlap(original: pd.DataFrame, competition: pd.DataFrame, column: str, decimals: int | None = None) -> dict:
    left = original[column]
    right = competition[column]
    if decimals is not None:
        left = pd.Series(np.round(left.to_numpy(dtype="float64"), decimals))
        right = pd.Series(np.round(right.to_numpy(dtype="float64"), decimals))
    elif pd.api.types.is_bool_dtype(left):
        left = left.astype("int8")
        right = right.astype("int8")
    original_values = set(left.dropna().unique().tolist())
    shared = original_values & set(right.dropna().unique().tolist())
    return {"original_unique": len(original_values), "shared_unique": len(shared),
            "fraction": float(len(shared) / len(original_values)) if original_values else None}


def audit(original: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> dict:
    original_hash = row_hashes(original, columns)
    train_hash = row_hashes(train, columns)
    test_hash = row_hashes(test, columns)
    train_labels: dict[int, set[int]] = {}
    for key, label in zip(train_hash, train[TARGET].astype(int).to_numpy()):
        train_labels.setdefault(int(key), set()).add(int(label))
    train_keys = set(train_labels)
    test_keys = {int(key) for key in test_hash}
    in_train = np.array([int(key) in train_keys for key in original_hash])
    in_test = np.array([int(key) in test_keys for key in original_hash])
    conflicts = 0
    for key, label in zip(original_hash[in_train], original.loc[in_train, TARGET].astype(int).to_numpy()):
        labels = train_labels[int(key)]
        if int(label) not in labels or len(labels) > 1:
            conflicts += 1
    overlaps = {column: column_overlap(original, train, column) for column in columns}
    for column in VALUE_COLUMNS:
        overlaps[f"{column}_rounded4"] = column_overlap(original, train, column, decimals=4)
    matched = in_train | in_test
    return {
        "source": f"https://www.kaggle.com/datasets/{DATASET}",
        "license": "CC0-1.0",
        "original_rows": int(len(original)),
        "original_fraud": int(original[TARGET].sum()),
        "feature_columns_aligned": columns,
        "exact_row_matches_train": int(in_train.sum()),
        "exact_row_matches_test": int(in_test.sum()),
        "exact_row_matches_any": int(matched.sum()),
        "exact_row_match_fraction": float(matched.mean()),
        "label_conflicts_among_train_matches": conflicts,
        "column_value_overlap_with_train": overlaps,
        "duplicate_stop": DUPLICATE_STOP,
        "stop": bool(matched.mean() >= DUPLICATE_STOP),
        "keep_mask": ~matched,
    }


def blend_with_cat(oof: pd.DataFrame) -> dict:
    cat = pd.read_parquet(CAT_OOF)
    merged = oof.merge(cat, on=ID, suffixes=("_xgb", "_cat"))
    if not np.array_equal(merged[f"{TARGET}_xgb"].to_numpy(), merged[f"{TARGET}_cat"].to_numpy()):
        raise ValueError("CatBoost 折外预测与本轮开发样本不一致")
    y = merged[f"{TARGET}_xgb"].to_numpy()
    xgb = merged["probability_xgb"].to_numpy()
    cat_probability = merged["probability_cat"].to_numpy()
    rows = []
    for weight in np.round(np.linspace(0, 1, 11), 1):
        probability = weight * xgb + (1 - weight) * cat_probability
        rows.append({"xgb_weight": float(weight), "cat_weight": float(1 - weight), "oof_auc": float(roc_auc_score(y, probability))})
    best = max(rows, key=lambda row: row["oof_auc"])
    return {"rows": rows, "best": best, "delta_vs_current_blend": best["oof_auc"] - CURRENT_BLEND_OOF_AUC}


def fit_folds(original: pd.DataFrame, competition: pd.DataFrame, keep: np.ndarray, columns: list[str], params: dict) -> dict:
    cfg = config()
    partition = splits(competition)
    dev_mask = partition.partition.to_numpy() == "dev"
    dev = competition.loc[dev_mask].reset_index(drop=True)
    folds = partition.loc[dev_mask, "fold"].to_numpy()
    extra = original.loc[keep].reset_index(drop=True)
    x_dev, y_dev = dev.drop(columns=[ID, TARGET]), dev[TARGET].to_numpy()
    folder = ROOT / "artifacts" / NAME
    folder.mkdir(parents=True, exist_ok=True)
    probabilities = np.full(len(dev), np.nan)
    fold_results = []
    started = pd.Timestamp.utcnow()
    for fold in range(cfg["n_splits"]):
        train_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        x_fit = pd.concat([x_dev.iloc[train_idx], extra.drop(columns=TARGET)], ignore_index=True)
        y_fit = np.concatenate([y_dev[train_idx], extra[TARGET].to_numpy()])
        bundle = fit_model("xgb", params, x_fit, y_fit, cfg["seed"] + fold, features=False)
        probability = predict(bundle, x_dev.iloc[valid_idx])
        probabilities[valid_idx] = probability
        fold_result = {"fold": fold, "competition_train_rows": int(len(train_idx)), "original_rows": int(len(extra)),
                       "valid_rows": int(len(valid_idx)), "training_device": bundle["training_device"],
                       "training_seconds": bundle["training_seconds"], "selected_iterations": bundle["selected_iterations"],
                       "metrics": metrics(y_dev[valid_idx], probability)}
        fold_results.append(fold_result)
        joblib.dump(bundle, folder / f"fold_{fold}.joblib", compress=3)
        write_json(folder / f"fold_{fold}.json", fold_result)
        print(f"{NAME} fold={fold} AUC={fold_result['metrics']['roc_auc']:.6f} seconds={bundle['training_seconds']:.1f} rounds={bundle['selected_iterations']}", flush=True)
        if fold == 0:
            write_json(folder / "feature_importance.json", feature_importance(bundle))
        del bundle
        gc.collect()
    if not np.isfinite(probabilities).all():
        raise RuntimeError("OOF 预测未覆盖所有开发样本")
    oof = pd.DataFrame({ID: dev[ID], "fold": folds, TARGET: y_dev, "probability": probabilities})
    oof.to_parquet(folder / "oof.parquet", index=False)
    threshold = f1_threshold(y_dev, probabilities)
    return {"started_at": str(started), "folds": fold_results, "oof_default": metrics(y_dev, probabilities),
            "oof_selected_threshold": metrics(y_dev, probabilities, threshold),
            "mean_fold_auc": float(np.mean([item["metrics"]["roc_auc"] for item in fold_results])),
            "std_fold_auc": float(np.std([item["metrics"]["roc_auc"] for item in fold_results], ddof=1)),
            "oof": oof}


def main() -> None:
    reference = json.loads((ROOT / "reports/experiments" / f"{REFERENCE_EXPERIMENT}.json").read_text(encoding="utf-8"))
    params = reference["spec"]["parameters"]
    if params.get("value_encoding") != "parts" or params.get("target_smooth") != 80.0:
        raise ValueError("参照实验已不是整数档、小数档、平滑 80")
    train, test = load_data()
    columns = feature_columns(train)
    original, extra_columns = load_original(columns)
    findings = audit(original, train, test, columns)
    keep = findings.pop("keep_mask")
    findings["dropped_columns"] = extra_columns
    findings["rows_used_for_training"] = int(keep.sum())
    print(f"整行重合 {findings['exact_row_matches_any']}/{findings['original_rows']}，保留 {findings['rows_used_for_training']} 行", flush=True)
    report = {"checked_at": timestamp(), "reference_experiment": REFERENCE_EXPERIMENT,
              "reference_oof_auc": reference["oof_default"]["roc_auc"],
              "current_blend_oof_auc": CURRENT_BLEND_OOF_AUC, "audit": findings, "holdout_used": False}
    if findings["stop"]:
        report["trained"] = False
        report["reason"] = "原始数据有一半以上的行与比赛训练或测试特征整行相同，停止训练"
        write_json(ROOT / "reports/original_augment.json", report)
        print(report["reason"], flush=True)
        return
    trained = fit_folds(original, train, keep, columns, params)
    oof = trained.pop("oof")
    single_auc = trained["oof_default"]["roc_auc"]
    blend = blend_with_cat(oof)
    report.update({"trained": True, "parameters": params, "result": trained, "blend_with_existing_cat": blend,
                   "clear_gain": CLEAR_GAIN,
                   "submission_created": blend["best"]["oof_auc"] >= CURRENT_BLEND_OOF_AUC + CLEAR_GAIN})
    report["deltas"] = {"versus_same_xgboost": single_auc - reference["oof_default"]["roc_auc"],
                        "versus_current_blend": blend["best"]["oof_auc"] - CURRENT_BLEND_OOF_AUC}
    write_json(ROOT / "reports/original_augment.json", report)
    print(f"单模型 OOF AUC={single_auc:.6f}，相对原 XGBoost {report['deltas']['versus_same_xgboost']:+.6f}", flush=True)
    print(f"与现有 CatBoost 融合 OOF AUC={blend['best']['oof_auc']:.6f}，相对当前融合 {report['deltas']['versus_current_blend']:+.6f}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
