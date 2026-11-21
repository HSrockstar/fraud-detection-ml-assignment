"""第二轮实验：实际吞吐量决策、完整三折比较与独立保存的网络模型。"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from .common import ID, ROOT, TARGET, config, load_data, splits, timestamp, write_json
from .evaluation import f1_threshold, metrics
from .neural import fit_network, predict_frame, restore_network

REPORTS = ROOT / "reports/round2"
ARTIFACTS = ROOT / "artifacts/round2"


def neural_config():
    return json.loads((ROOT / "configs/neural_round2.json").read_text(encoding="utf-8"))


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True).strip(),
            "packages": {name: importlib.metadata.version(name) for name in ["tabm", "rtdl-revisiting-models", "rtdl-num-embeddings", "numpy", "scikit-learn"]}}


def data():
    train, test = load_data()
    partition = splits(train)
    mask = partition.partition == "dev"
    dev = train.loc[mask].reset_index(drop=True)
    folds = partition.loc[mask, "fold"].to_numpy()
    return dev, folds, train, test


def save_bundle(path, bundle):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        torch.save(bundle, stream)


def load_bundle(path):
    with path.open("rb") as stream:
        return torch.load(stream, map_location="cpu", weights_only=False)


def benchmark():
    REPORTS.mkdir(parents=True, exist_ok=True)
    cfg = neural_config()
    dev, folds, _, _ = data()
    x, y = dev.drop(columns=[ID, TARGET]), dev[TARGET].to_numpy()
    outer_train = np.flatnonzero(folds != 0)
    inner, early = train_test_split(outer_train, test_size=config()["inner_early_stop_fraction"], stratify=y[outer_train], random_state=cfg["seed"])
    result = {"started_at": timestamp(), "environment": environment(), "configuration": cfg,
              "scope": "仅开发集第0训练折内部训练/早停数据；完整样本量短测。首轮留出集不参与。", "models": {}}
    for kind, params in cfg["models"].items():
        try:
            model, bundle = fit_network(kind, params, x.iloc[inner], y[inner], cfg["seed"], cfg["benchmark_epochs"],
                                         validation=(x.iloc[early], y[early]), patience=None,
                                         log_path=REPORTS / f"benchmark_{kind}_history.json")
            stable = bundle["history"][1:]
            epoch_seconds = float(np.mean([row["epoch_total_seconds"] for row in stable]))
            # 三折各做内部早停训练+全训练折重训，另计一个全部标签数据的最终模型。
            factor = 3 * (1 + len(outer_train)/len(inner)) + len(dev)/0.8/len(inner)
            estimated = (epoch_seconds * cfg["max_epochs"] * factor + bundle["setup_seconds"]*7) * 1.25
            result["models"][kind] = {"passed": True, "full_inner_train_rows": len(inner), "early_rows": len(early),
                                      "steady_epoch_seconds": epoch_seconds, "estimated_full_cv_and_final_minutes": estimated/60,
                                      "peak_memory_mib": bundle["peak_memory_mib"], "parameters": bundle["parameter_count"],
                                      "measured_epochs": bundle["history"], "estimate_max_epochs": cfg["max_epochs"]}
            result["models"][kind]["estimate_safety_factor"] = 1.25
            del model, bundle
            gc.collect(); torch.cuda.empty_cache()
        except (RuntimeError, torch.OutOfMemoryError) as exc:
            result["models"][kind] = {"passed": False, "error": str(exc)}
            torch.cuda.empty_cache()
        write_json(REPORTS / "hardware_benchmark.json", result)
    values = list(result["models"].values())
    all_passed = all(row["passed"] for row in values)
    suite_minutes = sum(row.get("estimated_full_cv_and_final_minutes", 0) for row in values)
    result["estimated_suite_minutes"] = suite_minutes if all_passed else None
    result["local_eligible"] = all_passed and suite_minutes <= cfg["local_suite_budget_minutes"] and all(row["estimated_full_cv_and_final_minutes"] <= cfg["local_model_budget_minutes"] for row in values)
    result["decision"] = "local"
    result["decision_reason"] = "用户指定全部训练使用本机 RTX 4060；预算估计仅供安排时间，不启用远程回退。"
    result["completed_at"] = timestamp()
    write_json(REPORTS / "hardware_benchmark.json", result)
    write_json(REPORTS / "environment.json", result["environment"])
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def run_cv(kind):
    cfg = neural_config()
    params = cfg["models"][kind]
    source_hash = hashlib.sha256((ROOT / "src/fraud/neural.py").read_bytes()).hexdigest()
    signature = {"configuration": cfg, "kind": kind, "source_sha256": source_hash, "versions": environment()["packages"], "torch": torch.__version__}
    path = REPORTS / f"{kind}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing["signature"] != signature:
            raise ValueError(f"{kind} 配置/代码/依赖已改变，请另存实验名称")
        print(f"复用第二轮已完成实验 {kind}", flush=True)
        return existing
    folder = ARTIFACTS / kind
    folder.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    signature_file = folder / "signature.json"
    if signature_file.exists() and json.loads(signature_file.read_text(encoding="utf-8")) != signature:
        raise ValueError(f"{kind} 未完成缓存的配置不一致")
    write_json(signature_file, signature)
    dev, folds, _, _ = data()
    x, y = dev.drop(columns=[ID, TARGET]), dev[TARGET].to_numpy()
    probabilities = np.full(len(dev), np.nan)
    fold_results = []
    start = time.perf_counter()
    for fold in range(config()["n_splits"]):
        outer_train, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        model_path, record_path = folder / f"fold_{fold}.pt", folder / f"fold_{fold}.json"
        if model_path.exists() and record_path.exists():
            bundle = load_bundle(model_path)
            model = restore_network(bundle)
            record = json.loads(record_path.read_text(encoding="utf-8"))
        else:
            inner, early = train_test_split(outer_train, test_size=config()["inner_early_stop_fraction"], stratify=y[outer_train], random_state=cfg["seed"]+fold)
            early_model, early_bundle = fit_network(kind, params, x.iloc[inner], y[inner], cfg["seed"]+fold, cfg["max_epochs"],
                                                     validation=(x.iloc[early], y[early]), patience=cfg["patience"], min_epochs=cfg["min_epochs"],
                                                     log_path=REPORTS / f"{kind}_fold{fold}_inner_history.json")
            selected_epochs = early_bundle["selected_epochs"]
            early_seconds = early_bundle["training_seconds"]
            del early_model, early_bundle
            gc.collect(); torch.cuda.empty_cache()
            model, bundle = fit_network(kind, params, x.iloc[outer_train], y[outer_train], cfg["seed"]+fold, selected_epochs,
                                        log_path=REPORTS / f"{kind}_fold{fold}_refit_history.json")
            save_bundle(model_path, bundle)
            record = {"fold": fold, "selected_epochs": selected_epochs, "inner_training_seconds": early_seconds,
                      "refit_training_seconds": bundle["training_seconds"], "peak_memory_mib": bundle["peak_memory_mib"],
                      "parameter_count": bundle["parameter_count"], "training_device": bundle["training_device"],
                      "outer_train_rows": len(outer_train), "valid_rows": len(valid)}
        probability = predict_frame(model, bundle["preprocessor"], x.iloc[valid], params["batch_size"]*2)
        probabilities[valid] = probability
        record["metrics"] = metrics(y[valid], probability)
        write_json(record_path, record)
        fold_results.append(record)
        print(f"{kind} fold={fold} OOF_AUC={record['metrics']['roc_auc']:.6f} AP={record['metrics']['average_precision']:.6f} epochs={record['selected_epochs']}", flush=True)
        del model, bundle
        gc.collect(); torch.cuda.empty_cache()
    if not np.isfinite(probabilities).all():
        raise RuntimeError("网络 OOF 覆盖不完整")
    threshold = f1_threshold(y, probabilities)
    pd.DataFrame({ID: dev[ID], TARGET: y, "fold": folds, "probability": probabilities}).to_parquet(folder / "oof.parquet", index=False)
    result = {"kind": kind, "signature": signature, "completed_at": timestamp(), "folds": fold_results,
              "oof_default": metrics(y, probabilities), "oof_selected_threshold": metrics(y, probabilities, threshold),
              "total_seconds": time.perf_counter()-start,
              "mean_fold_auc": float(np.mean([row["metrics"]["roc_auc"] for row in fold_results])),
              "std_fold_auc": float(np.std([row["metrics"]["roc_auc"] for row in fold_results], ddof=1)),
              "evaluation_scope": "第二轮开发集 OOF；首轮独立测试集已查看，本轮不重复用于调参或选型。"}
    write_json(path, result)
    return result
