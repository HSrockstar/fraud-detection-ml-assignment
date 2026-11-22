"""在开发集上把 CC0 公开折外预测掺进当前 7:3。权重不看公开榜，也不用 20% 留出集。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud.common import ID, ROOT, TARGET, load_data, splits, timestamp, write_json

SOURCE = ROOT / "data/external"
OURS = "ours_7_3"
MAX_MEMBERS = 4
MIN_GAIN = 0.00005


def rank01(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy() / (len(values) + 1.0)


def compositions(slots: int, units: int):
    if slots == 1:
        yield (units,)
        return
    for count in range(units + 1):
        for rest in compositions(slots - 1, units - count):
            yield (count,) + rest


def best_mix(y: np.ndarray, columns: list[np.ndarray], step: float) -> tuple[tuple[float, ...], float]:
    units = int(round(1.0 / step))
    best_weights = None
    best_auc = -1.0
    stacked = np.column_stack(columns)
    for counts in compositions(len(columns), units):
        weights = np.asarray(counts, dtype=np.float64) / units
        score = stacked @ weights
        auc = float(roc_auc_score(y, score))
        if auc > best_auc:
            best_auc = auc
            best_weights = tuple(float(weight) for weight in weights)
    return best_weights, best_auc


def main() -> None:
    manifest_path = next(SOURCE.rglob("oof_manifest.json"))
    folder = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train, test = load_data()
    split = splits(train)
    dev = split["partition"].eq("dev").to_numpy()
    y = train.loc[dev, TARGET].to_numpy()
    xgb = pd.read_parquet(ROOT / "artifacts/xgb_r2_parts_smooth80/oof.parquet")
    cat = pd.read_parquet(ROOT / "artifacts/cat_r2_parts/oof.parquet")
    ours = xgb.merge(cat, on=ID, suffixes=("_xgb", "_cat"))
    ours["probability"] = 0.7 * ours["probability_xgb"] + 0.3 * ours["probability_cat"]
    aligned = train.loc[dev, [ID]].merge(ours[[ID, "probability"]], on=ID, how="left")
    if aligned["probability"].isna().any():
        raise ValueError("当前 7:3 没有盖住全部开发集")
    dev_ranks = {OURS: rank01(aligned["probability"].to_numpy())}
    test_ranks = {}
    ours_test = pd.read_csv(ROOT / "submissions/xgb_cat_parts_70_30.csv")
    if not np.array_equal(ours_test[ID].to_numpy(), test[ID].to_numpy()):
        raise ValueError("当前提交的 id 顺序与测试集不一致")
    test_ranks[OURS] = rank01(ours_test[TARGET].to_numpy())
    rows = []
    for key, meta in manifest.items():
        oof = np.load(folder / f"oof_{key}.npy")
        predicted = np.load(folder / f"test_{key}.npy")
        if len(oof) != len(train) or len(predicted) != len(test):
            raise ValueError(f"{meta['name']} 的长度与官方文件不一致")
        if not np.isfinite(oof).all() or not np.isfinite(predicted).all():
            raise ValueError(f"{meta['name']} 含有非有限数")
        dev_rank = rank01(oof[dev])
        auc = float(roc_auc_score(y, dev_rank))
        spearman = float(pd.Series(dev_rank).corr(pd.Series(dev_ranks[OURS]), method="spearman"))
        dev_ranks[meta["name"]] = dev_rank
        test_ranks[meta["name"]] = rank01(predicted)
        rows.append({"name": meta["name"], "key": key, "published_oof_auc": meta.get("auc"), "dev_auc": auc, "spearman_vs_ours": spearman})
    ours_auc = float(roc_auc_score(y, dev_ranks[OURS]))
    pool = [row["name"] for row in sorted(rows, key=lambda row: row["dev_auc"], reverse=True)[:8]]
    selected = [OURS]
    current_weights, current_auc = (1.0,), ours_auc
    history = [{"members": [OURS], "weights": [1.0], "dev_auc": ours_auc}]
    while len(selected) < MAX_MEMBERS:
        best = None
        step = 0.05 if len(selected) == 1 else 0.1
        for name in pool:
            if name in selected:
                continue
            weights, auc = best_mix(y, [dev_ranks[item] for item in selected] + [dev_ranks[name]], step)
            if best is None or auc > best[0]:
                best = (auc, name, weights)
        if best is None or best[0] - current_auc < MIN_GAIN:
            break
        current_auc, added, current_weights = best
        selected.append(added)
        history.append({"members": list(selected), "weights": list(current_weights), "dev_auc": current_auc})
    blended = sum(weight * test_ranks[name] for weight, name in zip(current_weights, selected))
    submission = pd.DataFrame({ID: test[ID], TARGET: blended})
    if not np.isfinite(blended).all() or ((blended <= 0) | (blended >= 1)).any():
        raise ValueError("融合名次超出 (0, 1)")
    output = ROOT / "submissions/public_oof_blend.csv"
    submission.to_csv(output, index=False)
    summary = {
        "source": "https://www.kaggle.com/datasets/alexchilton/s6e11-oof-predictions",
        "license": "CC0-1.0",
        "author_note": "作者写明这些折外预测可以用来融合，不必再训练。行顺序与官方 train.csv、test.csv 一致。",
        "selection": "只在我们的开发集上比较名次平均。20% 留出集和公开榜都不参与选权重。",
        "ours_dev_auc": ours_auc,
        "members_scored": len(rows),
        "strongest_public_on_our_dev": max(rows, key=lambda row: row["dev_auc"]),
        "selected_members": selected,
        "selected_weights": list(current_weights),
        "selected_dev_auc": current_auc,
        "delta_vs_ours": current_auc - ours_auc,
        "history": history,
        "public_members": sorted(rows, key=lambda row: row["dev_auc"], reverse=True),
        "rows": len(submission),
        "file": "submissions/public_oof_blend.csv",
        "created_at": timestamp(),
    }
    write_json(ROOT / "reports/public_oof_blend.json", summary)
    print(json.dumps({
        "ours_dev_auc": ours_auc,
        "selected_members": selected,
        "selected_weights": list(current_weights),
        "selected_dev_auc": current_auc,
        "delta_vs_ours": current_auc - ours_auc,
        "strongest_public_dev_auc": summary["strongest_public_on_our_dev"]["dev_auc"],
        "strongest_public_name": summary["strongest_public_on_our_dev"]["name"],
        "strongest_public_spearman": summary["strongest_public_on_our_dev"]["spearman_vs_ours"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
