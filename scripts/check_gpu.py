"""实际训练两种模型，核验 GPU 执行和运行依赖。"""
from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud.common import ROOT, timestamp, write_json


def main():
    import numpy as np
    from catboost import CatBoostClassifier
    from sklearn.datasets import make_classification
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    x, y = make_classification(n_samples=8000, n_features=16, n_informative=10, weights=[0.9, 0.1], random_state=42)
    xt, xv, yt, yv = train_test_split(x.astype(np.float32), y, stratify=y, random_state=42)
    checks = {}
    for name, model in {
        "xgboost": XGBClassifier(n_estimators=100, max_depth=4, tree_method="hist", device="cuda", n_jobs=4, eval_metric="auc", random_state=42),
        "catboost": CatBoostClassifier(iterations=100, depth=4, task_type="GPU", devices="0", verbose=False, allow_writing_files=False, random_seed=42),
    }.items():
        start = time.perf_counter()
        model.fit(xt, yt)
        device = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"] if name == "xgboost" else model.get_all_params()["task_type"]
        if not (str(device).startswith("cuda") or device == "GPU"):
            raise RuntimeError(f"{name} 未使用 GPU: {device}")
        auc = roc_auc_score(yv, model.predict_proba(xv)[:, 1])
        if auc < 0.8:
            raise RuntimeError(f"{name} GPU 训练预测自检异常，AUC={auc}")
        checks[name] = {"device": device, "auc": auc, "seconds": time.perf_counter() - start}
        print(name, checks[name], flush=True)
    result = {
        "checked_at": timestamp(), "python": platform.python_version(), "platform": platform.platform(),
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], encoding="utf-8").strip(),
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "xgboost", "catboost", "optuna", "matplotlib", "kaggle"]},
        "training_checks": checks,
    }
    write_json(ROOT / "reports/environment.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
