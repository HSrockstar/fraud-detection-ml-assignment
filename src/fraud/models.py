from __future__ import annotations

import json
import time

import numpy as np
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from .common import config
from .features import cat_frame, engineer, preprocessor, value_keys

RESERVED_PARAMS = {"target_encoding", "value_encoding", "target_smooth"}


def model_frame(x, features, params):
    frame = engineer(x, features)
    mode = params.get("value_encoding")
    if mode:
        frame = value_keys(frame, mode)
    return frame


def fit_model(kind, params, x, y, seed, features=False, fixed_iterations=None):
    """外层验证样本不参与早停；轮数由训练数据内部的独立子集确定。"""
    cfg = config()
    x = model_frame(x, features, params)
    y = np.asarray(y)
    start = time.perf_counter()
    selected_iterations = None
    prep = None
    model = None
    trained_device = "CPU"
    with threadpool_limits(limits=6):
        if kind in {"logistic", "tree"}:
            prep = preprocessor(x, scale=kind == "logistic")
            encoded = prep.fit_transform(x).astype("float32")
            model = (LogisticRegression(C=1.0, max_iter=1000, tol=1e-5, **params) if kind == "logistic" else
                     DecisionTreeClassifier(max_depth=8, min_samples_leaf=100, class_weight="balanced", random_state=seed, **params))
            model.fit(encoded, y)
        else:
            train_idx, early_idx = train_test_split(np.arange(len(y)), test_size=cfg["inner_early_stop_fraction"], stratify=y, random_state=seed)
            if kind == "xgb":
                xgb_params = {key: value for key, value in params.items() if key not in RESERVED_PARAMS}
                use_target_encoding = bool(params.get("target_encoding", False))
                target_smooth = float(params.get("target_smooth", 20.0))
                common = dict(tree_method="hist", device="cuda", objective="binary:logistic", eval_metric="auc", n_jobs=6, random_state=seed)
                if fixed_iterations is None:
                    early_prep = preprocessor(x, target_encode=use_target_encoding, seed=seed, target_smooth=target_smooth)
                    inner_x = early_prep.fit_transform(x.iloc[train_idx], y[train_idx]).astype("float32")
                    early_x = early_prep.transform(x.iloc[early_idx]).astype("float32")
                    early_model = XGBClassifier(**common, **xgb_params, early_stopping_rounds=cfg["early_stopping_rounds"])
                    early_model.fit(inner_x, y[train_idx], eval_set=[(early_x, y[early_idx])], verbose=False)
                    selected_iterations = int(early_model.best_iteration + 1)
                    del early_model, early_prep, inner_x, early_x
                else:
                    selected_iterations = int(fixed_iterations)
                prep = preprocessor(x, target_encode=use_target_encoding, seed=seed, target_smooth=target_smooth)
                encoded = prep.fit_transform(x, y).astype("float32")
                final_params = {**xgb_params, "n_estimators": selected_iterations}
                model = XGBClassifier(**common, **final_params)
                model.fit(encoded, y, verbose=False)
                trained_device = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
                if not trained_device.startswith("cuda"):
                    raise RuntimeError("XGBoost 未在 CUDA 上训练，停止实验")
                # 输入数据位于 CPU；小批量预测切回 CPU，训练仍强制使用 GPU。
                model.set_params(device="cpu")
            elif kind == "cat":
                encoded, cat_columns = cat_frame(x)
                common = dict(task_type="GPU", devices="0", loss_function="Logloss", eval_metric="AUC", random_seed=seed, thread_count=6, verbose=False, allow_writing_files=False)
                if fixed_iterations is None:
                    early_model = CatBoostClassifier(**common, **params)
                    early_model.fit(encoded.iloc[train_idx], y[train_idx], cat_features=cat_columns,
                                    eval_set=(encoded.iloc[early_idx], y[early_idx]),
                                    early_stopping_rounds=cfg["early_stopping_rounds"], use_best_model=True)
                    selected_iterations = int(early_model.tree_count_)
                    del early_model
                else:
                    selected_iterations = int(fixed_iterations)
                final_params = {**params, "iterations": selected_iterations}
                model = CatBoostClassifier(**common, **final_params)
                model.fit(encoded, y, cat_features=cat_columns)
                trained_device = model.get_all_params()["task_type"]
                if trained_device != "GPU":
                    raise RuntimeError("CatBoost 未在 GPU 上训练，停止实验")
            else:
                raise ValueError(f"未知模型: {kind}")
    return {"kind": kind, "model": model, "preprocessor": prep, "features": features,
            "parameters": params, "selected_iterations": selected_iterations,
            "training_device": trained_device, "training_seconds": time.perf_counter() - start}


def predict(bundle, x):
    x = model_frame(x, bundle["features"], bundle["parameters"])
    encoded = cat_frame(x)[0] if bundle["kind"] == "cat" else bundle["preprocessor"].transform(x).astype("float32")
    with threadpool_limits(limits=6):
        return bundle["model"].predict_proba(encoded)[:, 1]


def feature_importance(bundle):
    if bundle["kind"] == "cat":
        return dict(zip(bundle["model"].feature_names_, bundle["model"].feature_importances_.tolist()))
    if bundle["kind"] == "logistic":
        values = np.abs(bundle["model"].coef_[0])
    else:
        values = bundle["model"].feature_importances_
    return dict(zip(bundle["preprocessor"].get_feature_names_out().tolist(), values.tolist()))
