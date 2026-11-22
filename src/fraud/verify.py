from __future__ import annotations

import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from .common import ID, ROOT, TARGET, load_data, splits, timestamp, write_json
from .evaluation import metrics
from .models import predict


def run():
    train, test = load_data()
    partition = splits(train)
    dev_mask = partition.partition == "dev"
    dev = train.loc[dev_mask].reset_index(drop=True)
    folds = partition.loc[dev_mask, "fold"].to_numpy()
    columns = train.columns.drop([ID, TARGET]).tolist()
    checks = []
    manifest = json.loads((ROOT / "reports/data_manifest.json").read_text(encoding="utf-8"))
    for name, value in manifest["files"].items():
        with (ROOT / "data/raw" / name).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        assert digest == value["sha256"], f"原始文件发生变化: {name}"
    checks.append("三个官方数据文件 SHA256 与下载清单一致")
    selection = json.loads((ROOT / "reports/selection.json").read_text(encoding="utf-8"))
    for name in selection["all_experiments"]:
        path = ROOT / "reports/experiments" / f"{name}.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["spec"]["name"] == name
        oof = pd.read_parquet(ROOT / "artifacts" / name / "oof.parquet")
        assert np.array_equal(oof[ID], dev[ID]) and np.array_equal(oof.fold, folds)
        assert np.isclose(metrics(oof[TARGET], oof.probability)["roc_auc"], result["oof_default"]["roc_auc"], atol=1e-12)
        sample = np.flatnonzero(folds == 0)[:100]
        bundle = joblib.load(ROOT / "artifacts" / name / "fold_0.joblib")
        assert np.allclose(predict(bundle, dev.iloc[sample][columns]), oof.iloc[sample].probability, atol=1e-6)
        if result["spec"]["kind"] in {"xgb", "cat"}:
            assert all(f["training_device"] in ["cuda:0", "GPU"] for f in result["folds"])
        checks.append(f"{name}: OOF 覆盖、指标复算、持久化推理一致")
    holdout = json.loads((ROOT / "reports/holdout_metrics.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(ROOT / "artifacts/evaluation/holdout_predictions.parquet")
    assert set(frame[ID]).isdisjoint(dev[ID])
    assert np.array_equal(frame[ID], train.loc[~dev_mask, ID])
    for name in ["selected"] + selection["all_experiments"]:
        recomputed = metrics(frame[TARGET], frame[name], holdout[name]["oof_threshold"]["threshold"])
        assert all(np.isclose(recomputed[k], holdout[name]["oof_threshold"][k]) for k in ["roc_auc", "average_precision", "precision", "recall", "f1"])
    checks.append("独立测试集与开发集完全分离，所有最终指标复算一致")
    sample = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    submission = pd.read_csv(ROOT / "submissions/round1_best.csv")
    assert submission.shape == sample.shape and submission.columns.tolist() == [ID, TARGET]
    assert np.array_equal(submission[ID], sample[ID]) and np.array_equal(submission[ID], test[ID])
    assert np.isfinite(submission[TARGET]).all() and submission[TARGET].between(0, 1).all()
    expected = np.zeros(100)
    for name, weight in selection["weights"].items():
        bundle = joblib.load(ROOT / "artifacts/final" / f"{name}.joblib")
        expected += predict(bundle, test[columns].iloc[:100]) * weight
    assert np.allclose(expected, submission[TARGET].iloc[:100], atol=1e-6)
    checks.append("299984 行提交文件列名、顺序、范围和最终模型推理一致")
    for name in ["实验报告.md", "report_content.json", "model_comparison.csv", "environment.json"]:
        assert (ROOT / "reports" / name).is_file()
    if (ROOT / "reports/extensions/results.json").exists():
        from .extensions import verify as verify_extensions
        checks.append(verify_extensions(dev, folds))
    write_json(ROOT / "reports/verification.json", {"checked_at": timestamp(), "passed": True, "checks": checks})
    print("\n".join(checks), flush=True)
