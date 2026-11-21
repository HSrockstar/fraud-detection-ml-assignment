from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[2]
TARGET = "is_fraud"
ID = "id"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def config() -> dict:
    return json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = ROOT / "data/raw"
    train, test = pd.read_csv(raw / "train.csv"), pd.read_csv(raw / "test.csv")
    if not set(train[TARGET].unique()) == {0, 1} or train[TARGET].isna().any():
        raise ValueError("训练标签必须是无缺失的二分类 0/1")
    if not train[ID].is_unique or not test[ID].is_unique:
        raise ValueError("样本 id 重复")
    if set(train[ID]) & set(test[ID]):
        raise ValueError("训练集和比赛测试集 id 重叠")
    if list(train.drop(columns=TARGET).columns) != list(test.columns):
        raise ValueError("训练与测试特征列不一致")
    return train, test


def splits(train: pd.DataFrame) -> pd.DataFrame:
    cfg = config()
    path = ROOT / "artifacts/splits.parquet"
    # 标签和 id 的指纹防止误用其他数据或重排后的划分。
    digest = hashlib.sha256(pd.util.hash_pandas_object(train[[ID, TARGET]], index=True).values.tobytes()).hexdigest()
    meta_path = path.with_suffix(".json")
    if path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = {"sha256": digest, "seed": cfg["seed"], "holdout_fraction": cfg["holdout_fraction"], "n_splits": cfg["n_splits"]}
        if any(meta.get(k) != v for k, v in expected.items()):
            raise ValueError("数据或划分配置变化，请另存已有实验后重新生成划分")
        result = pd.read_parquet(path)
        if not np.array_equal(result[ID].to_numpy(), train[ID].to_numpy()):
            raise ValueError("划分顺序与训练数据不一致")
        return result
    dev, holdout = train_test_split(np.arange(len(train)), test_size=cfg["holdout_fraction"], stratify=train[TARGET], random_state=cfg["seed"])
    result = pd.DataFrame({ID: train[ID], "partition": "dev", "fold": -1})
    result.loc[holdout, "partition"] = "holdout"
    cv = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"])
    for fold, (_, valid) in enumerate(cv.split(dev, train.iloc[dev][TARGET])):
        result.loc[dev[valid], "fold"] = fold
    assert (result.loc[dev, "fold"] >= 0).all()
    assert (result.loc[holdout, "fold"] == -1).all()
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)
    write_json(meta_path, {"sha256": digest, "seed": cfg["seed"], "holdout_fraction": cfg["holdout_fraction"], "n_splits": cfg["n_splits"], "created_at": timestamp(), "dev_rows": len(dev), "holdout_rows": len(holdout)})
    return result


def plot_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False, "figure.dpi": 120, "savefig.dpi": 180})
    (ROOT / "reports/figures").mkdir(parents=True, exist_ok=True)
    return plt
