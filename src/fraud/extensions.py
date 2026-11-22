"""开发集三折上的手写逻辑回归对照和单因素稳健性实验。"""

import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from .common import ID, ROOT, TARGET, config, timestamp, write_json
from .experiments import development_data
from .features import preprocessor
from .models import fit_model, predict
from .scratch_logistic import ScratchLogisticRegression, objective

FRACTIONS = (0.10, 0.25, 0.50, 1.0)
NOISE = (0.10, 0.25, 0.50)
OUTLIERS = (0.01, 0.05, 0.10)
CONTINUOUS = (
    "amount_usd", "account_balance_usd", "distance_from_home_km",
    "hours_since_last_txn", "velocity_score", "merchant_risk_score",
)
SCORES = ("roc_auc", "average_precision", "log_loss")


def scores(labels, probability):
    return {
        "roc_auc": float(roc_auc_score(labels, probability)),
        "average_precision": float(average_precision_score(labels, probability)),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
    }


def nested_subsets(labels, seed):
    rng = np.random.default_rng(seed)
    orders = [rng.permutation(np.flatnonzero(labels == label)) for label in (0, 1)]
    for fraction in FRACTIONS:
        # 每类复用同一排列的前缀，让小样本集嵌套于大样本集。
        parts = [order[:max(1, round(len(order) * fraction))] for order in orders]
        yield fraction, np.sort(np.concatenate(parts))


def perturbed_inputs(train, validation, seed):
    fields = list(CONTINUOUS)
    quantiles = train[fields].quantile([0.25, 0.75])
    q3 = quantiles.loc[0.75].to_numpy()
    iqr = q3 - quantiles.loc[0.25].to_numpy()
    rng = np.random.default_rng(seed)
    clean = validation[fields].to_numpy(dtype=np.float64)
    noise = rng.standard_normal(clean.shape) * iqr
    for sigma in NOISE:
        altered = validation.copy()
        altered[fields] = clean + sigma * noise
        yield f"noise_{sigma:.2f}", altered

    order = rng.permutation(len(validation))
    columns = rng.integers(len(fields), size=len(validation))
    for fraction in OUTLIERS:
        rows = order[:round(len(validation) * fraction)]
        features = columns[:len(rows)]
        values = clean.copy()
        values[rows, features] = (q3 + 10.0 * iqr)[features]
        altered = validation.copy()
        altered[fields] = values
        yield f"outliers_{fraction:.2f}", altered


def transform(prep, frame):
    return np.ascontiguousarray(prep.transform(frame), dtype=np.float64)


def compare_logistic(x_train, y_train, x_valid):
    prep = preprocessor(x_train, scale=True).fit(x_train)
    train = transform(prep, x_train)
    valid = transform(prep, x_valid)
    fitted = {}
    for name, model in (
        ("scratch", ScratchLogisticRegression()),
        ("sklearn", LogisticRegression(C=1.0, solver="lbfgs", tol=1e-10, max_iter=1000)),
    ):
        started = time.perf_counter()
        model.fit(train, y_train)
        fitted[name] = (model, time.perf_counter() - started)
    scratch, scratch_time = fitted["scratch"]
    sklearn, sklearn_time = fitted["sklearn"]
    design = np.column_stack([train, np.ones(len(train))])
    p_scratch, p_sklearn = scratch.predict_proba(valid)[:, 1], sklearn.predict_proba(valid)[:, 1]
    loss_scratch = objective(design, y_train, np.r_[scratch.coef_, scratch.intercept_], 1.0 / len(train))
    loss_sklearn = objective(design, y_train, np.r_[sklearn.coef_[0], sklearn.intercept_[0]], 1.0 / len(train))
    detail = {
        "scratch_fit_seconds": scratch_time,
        "sklearn_fit_seconds": sklearn_time,
        "scratch_iterations": scratch.n_iter_,
        "sklearn_iterations": int(sklearn.n_iter_[0]),
        "scratch_gradient_inf_norm": scratch.gradient_norm_,
        "training_objective_scratch": loss_scratch,
        "training_objective_sklearn": loss_sklearn,
        "max_absolute_probability_difference": float(np.max(np.abs(p_scratch - p_sklearn))),
    }
    return p_scratch, p_sklearn, prep, sklearn, detail


def conditions():
    return [
        *(f"size_{fraction:.2f}" for fraction in FRACTIONS),
        *(f"noise_{sigma:.2f}" for sigma in NOISE),
        *(f"outliers_{fraction:.2f}" for fraction in OUTLIERS),
    ]


def run():
    reports = ROOT / "reports/extensions"
    artifacts = ROOT / "artifacts/extensions"
    result_path = reports / "results.json"
    prediction_path = artifacts / "oof.parquet"
    cfg = config()
    source_files = [
        "extensions.py", "scratch_logistic.py", "features.py", "models.py", "common.py",
    ]
    fingerprint = {name: hashlib.sha256((ROOT / "src/fraud" / name).read_bytes()).hexdigest() for name in source_files}
    with (ROOT / "data/raw/train.csv").open("rb") as source:
        fingerprint["train.csv"] = hashlib.file_digest(source, "sha256").hexdigest()
    if result_path.exists() and prediction_path.exists():
        old = json.loads(result_path.read_text(encoding="utf-8"))
        if old["fingerprint"] == fingerprint and old["config"] == cfg:
            print("复用已完成的手写逻辑回归和稳健性实验。", flush=True)
            return

    dev, folds, _ = development_data()
    features, labels = dev.drop(columns=[ID, TARGET]), dev[TARGET].to_numpy()
    robustness = [f"{condition}_{name}" for condition in conditions() for name in ("logistic", "xgb")]
    probabilities = {name: np.full(len(dev), np.nan) for name in ["scratch", "sklearn", *robustness]}
    details = []
    for fold in range(cfg["n_splits"]):
        train = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        x_train, y_train, x_valid = features.iloc[train], labels[train], features.iloc[valid]
        seed = cfg["seed"] + fold
        with threadpool_limits(limits=6):
            p_scratch, p_sklearn, prep, logistic, detail = compare_logistic(x_train, y_train, x_valid)
            probabilities["scratch"][valid], probabilities["sklearn"][valid] = p_scratch, p_sklearn
            xgb = fit_model("xgb", cfg["xgb_baseline"], x_train, y_train, seed)
            sizes = {}
            for fraction, local in nested_subsets(y_train, seed):
                key = f"size_{fraction:.2f}"
                sizes[key] = {"rows": len(local), "positives": int(y_train[local].sum())}
                if fraction == 1.0:
                    probabilities[f"{key}_logistic"][valid] = p_sklearn
                    experiment_xgb = xgb
                else:
                    subset = x_train.iloc[local]
                    subset_prep = preprocessor(subset, scale=True).fit(subset)
                    subset_logistic = LogisticRegression(C=1.0, solver="lbfgs", tol=1e-10, max_iter=1000)
                    subset_logistic.fit(transform(subset_prep, subset), y_train[local])
                    probabilities[f"{key}_logistic"][valid] = subset_logistic.predict_proba(transform(subset_prep, x_valid))[:, 1]
                    experiment_xgb = fit_model("xgb", cfg["xgb_baseline"], subset, y_train[local], seed)
                probabilities[f"{key}_xgb"][valid] = predict(experiment_xgb, x_valid)
                sizes[key]["xgb_rounds"] = experiment_xgb["selected_iterations"]
                print(f"第 {fold + 1} 折，完成 {fraction:.0%} 样本量对照。", flush=True)
            for key, altered in perturbed_inputs(x_train, x_valid, seed + 1000):
                probabilities[f"{key}_logistic"][valid] = logistic.predict_proba(transform(prep, altered))[:, 1]
                probabilities[f"{key}_xgb"][valid] = predict(xgb, altered)
        details.append({"fold": fold, "valid_rows": len(valid), "sizes": sizes, **detail})
        print(f"第 {fold + 1} 折，完成噪声和异常值对照。", flush=True)

    if not all(np.isfinite(column).all() for column in probabilities.values()):
        raise RuntimeError("扩展实验的折外预测覆盖不完整")
    all_scores = {name: scores(labels, probability) for name, probability in probabilities.items()}
    fold_scores = {
        name: [scores(labels[folds == fold], probability[folds == fold]) for fold in range(cfg["n_splits"])]
        for name, probability in probabilities.items()
    }
    artifacts.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: dev[ID], TARGET: labels, "fold": folds, **probabilities}).to_parquet(prediction_path, index=False)
    write_json(result_path, {
        "completed_at": timestamp(),
        "fingerprint": fingerprint,
        "config": cfg,
        "protocol": {
            "scope": "同一固定开发集三折 OOF，保留的独立测试集不参与",
            "sample_fractions": FRACTIONS,
            "noise_sigmas_in_training_iqr": NOISE,
            "outlier_row_fractions": OUTLIERS,
            "outlier_replacement": "每个受扰行的 1 列替换为当折训练 Q3 + 10×IQR",
            "perturbed_features": CONTINUOUS,
        },
        "folds": details,
        "oof": all_scores,
        "fold_scores": fold_scores,
        "mean_absolute_probability_difference": float(np.abs(probabilities["scratch"] - probabilities["sklearn"]).mean()),
    })
    print("手写逻辑回归和稳健性实验结果已保存。", flush=True)


def verify(dev, folds):
    result = json.loads((ROOT / "reports/extensions/results.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(ROOT / "artifacts/extensions/oof.parquet")
    expected = {"scratch", "sklearn", *(f"{key}_{name}" for key in conditions() for name in ("logistic", "xgb"))}
    assert set(result["oof"]) == expected
    assert set(frame.columns) == expected | {ID, TARGET, "fold"}
    assert np.array_equal(frame[ID], dev[ID]) and np.array_equal(frame["fold"], folds)
    assert np.array_equal(frame[TARGET], dev[TARGET])
    assert frame["sklearn"].equals(frame["size_1.00_logistic"])
    assert abs(frame["scratch"] - frame["sklearn"]).mean() < 1e-4
    for name, recorded in result["oof"].items():
        assert frame[name].between(0.0, 1.0).all()
        current = scores(frame[TARGET], frame[name])
        assert all(np.isclose(current[key], recorded[key], rtol=0, atol=1e-12) for key in SCORES), name
        for fold, metrics in enumerate(result["fold_scores"][name]):
            rows = frame["fold"] == fold
            actual = scores(frame.loc[rows, TARGET], frame.loc[rows, name])
            assert all(np.isclose(actual[key], metrics[key], rtol=0, atol=1e-12) for key in SCORES), (name, fold)
    return "手写逻辑回归与鲁棒性实验的折外覆盖、训练集口径和全部场景指标复算一致"
