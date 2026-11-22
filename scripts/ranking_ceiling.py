"""检查当前 7:3 融合在开发集上的排序上限。只读已有折外预测，不训练，不看独立测试集。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud.common import ID, ROOT, TARGET, load_data, splits, timestamp, write_json

XGB_NAME = "xgb_r2_parts_smooth80"
CAT_NAME = "cat_r2_parts"
XGB_WEIGHT = 0.7
REPORT_JSON = ROOT / "reports" / "ranking_ceiling.json"
REPORT_MD = ROOT / "reports" / "ranking_ceiling.md"


def pairwise_win_rate(positive: np.ndarray, negative: np.ndarray) -> float:
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    ordered = np.sort(negative)
    less = np.searchsorted(ordered, positive, side="left")
    equal = np.searchsorted(ordered, positive, side="right") - less
    return float((less.sum() + 0.5 * equal.sum()) / (len(positive) * len(negative)))


def safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2 or len(y) < 2:
        return None
    if np.nanstd(score) == 0:
        return 0.5
    return float(roc_auc_score(y, score))


def oriented_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    forward = safe_auc(y, score)
    backward = safe_auc(y, -score)
    if forward is None or backward is None:
        return None
    return max(forward, backward)


def leave_one_out_rate(y: pd.Series, key: pd.Series) -> pd.Series:
    grouped = y.groupby(key, sort=False)
    total = grouped.transform("size")
    frauds = grouped.transform("sum")
    rate = (frauds - y) / (total - 1)
    return rate.where(total > 1, 0.5)


def group_rate(y: pd.Series, key: pd.Series) -> pd.Series:
    return y.groupby(key, sort=False).transform("mean")


def weighted_within_auc(frame: pd.DataFrame, key: pd.Series) -> dict:
    weighted_sum = 0.0
    weight_total = 0
    both = 0
    for _, part in frame.groupby(key, sort=False):
        y = part[TARGET].to_numpy()
        frauds = int(y.sum())
        normals = len(part) - frauds
        if frauds == 0 or normals == 0:
            continue
        both += 1
        weight = frauds * normals
        weighted_sum += roc_auc_score(y, part["p"].to_numpy()) * weight
        weight_total += weight
    if weight_total == 0:
        return {"patterns_with_both_classes": 0, "pair_weight": 0, "weighted_auc": None}
    return {
        "patterns_with_both_classes": both,
        "pair_weight": int(weight_total),
        "weighted_auc": float(weighted_sum / weight_total),
    }


def quantiles(series: pd.Series) -> dict:
    points = series.quantile([0.1, 0.5, 0.9, 0.99]).to_dict()
    return {f"p{int(q * 100)}": float(value) for q, value in points.items()}


def flag_frame(dev: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "cvv_retry": dev["cvv_retry_count"].ge(1),
        "no_auth": dev["auth_method"].eq("No Authentication"),
        "new_merchant": dev["is_new_merchant"].astype(bool),
        "ip_mismatch": dev["ip_country_mismatch"].astype(bool),
        "billing_mismatch": dev["billing_shipping_mismatch"].astype(bool),
        "foreign": dev["is_foreign_transaction"].astype(bool),
        "vpn": dev["used_vpn"].astype(bool),
        "ai_flag": dev["is_ai_generated_scam_attempt"].astype(bool),
        "gift_crypto_game": dev["merchant_category"].isin(["Gift Cards", "Crypto Exchange", "Gaming"]),
        "risk_ge_60": dev["merchant_risk_score"].ge(60),
        "velocity_ge_40": dev["velocity_score"].ge(40),
        "txn_ge_6": dev["txn_count_last_24h"].ge(6),
    })


def zone_record(frame: pd.DataFrame) -> dict:
    y = frame[TARGET].to_numpy()
    fraud = frame.loc[frame[TARGET].eq(1), "p"]
    normal = frame.loc[frame[TARGET].eq(0), "p"]
    return {
        "rows": int(len(frame)),
        "frauds": int(y.sum()),
        "fraud_rate": float(y.mean()) if len(frame) else None,
        "auc": safe_auc(y, frame["p"].to_numpy()),
        "fraud_score": quantiles(fraud) if len(fraud) else {},
        "normal_score": quantiles(normal) if len(normal) else {},
    }


def group_profile(frame: pd.DataFrame, flags: pd.DataFrame) -> dict:
    profile = {
        "rows": int(len(frame)),
        "score": quantiles(frame["p"]),
        "flag_count_median": float(flags.loc[frame.index].sum(axis=1).median()),
        "merchant_risk_score_median": float(frame["merchant_risk_score"].median()),
        "velocity_score_median": float(frame["velocity_score"].median()),
        "customer_age_median": float(frame["customer_age"].median()),
        "amount_usd_median": float(frame["amount_usd"].median()),
    }
    rates = flags.loc[frame.index].mean()
    profile["flag_share"] = {name: float(rate) for name, rate in rates.items()}
    return profile


def build_frame() -> pd.DataFrame:
    train, _ = load_data()
    partition = splits(train)
    dev = train.loc[partition["partition"].eq("dev")].copy()
    xgb = pd.read_parquet(ROOT / "artifacts" / XGB_NAME / "oof.parquet")
    cat = pd.read_parquet(ROOT / "artifacts" / CAT_NAME / "oof.parquet")
    merged = xgb.merge(cat, on=ID, suffixes=("_xgb", "_cat"))
    if not np.array_equal(merged[f"{TARGET}_xgb"], merged[f"{TARGET}_cat"]):
        raise ValueError("两份折外预测的标签不一致")
    dev = dev.merge(
        merged[[ID, "probability_xgb", "probability_cat"]],
        on=ID,
        how="inner",
        validate="one_to_one",
    )
    if len(dev) != int(partition["partition"].eq("dev").sum()):
        raise ValueError("折外预测没有覆盖全部开发集")
    dev["p"] = XGB_WEIGHT * dev["probability_xgb"] + (1 - XGB_WEIGHT) * dev["probability_cat"]
    if dev["p"].isna().any() or not dev["p"].between(0, 1).all():
        raise ValueError("融合概率无效")
    return dev


def decomposition(dev: pd.DataFrame, loud: pd.Series) -> dict:
    fraud = dev[TARGET].eq(1)
    pieces = {
        "loud_fraud_vs_quiet_normal": (fraud & loud, ~fraud & ~loud),
        "loud_fraud_vs_loud_normal": (fraud & loud, ~fraud & loud),
        "quiet_fraud_vs_quiet_normal": (fraud & ~loud, ~fraud & ~loud),
        "quiet_fraud_vs_loud_normal": (fraud & ~loud, ~fraud & loud),
    }
    total_pairs = int(fraud.sum() * (~fraud).sum())
    rows = []
    covered = 0.0
    for name, (pos_mask, neg_mask) in pieces.items():
        positive = dev.loc[pos_mask, "p"].to_numpy()
        negative = dev.loc[neg_mask, "p"].to_numpy()
        pairs = int(len(positive) * len(negative))
        win = pairwise_win_rate(positive, negative)
        share = pairs / total_pairs
        covered += share * win
        rows.append({
            "pair": name,
            "positive": int(len(positive)),
            "negative": int(len(negative)),
            "pair_share": share,
            "win_rate": win,
            "auc_contribution": share * win,
        })
    return {"total_pairs": total_pairs, "reconstructed_auc": covered, "pieces": rows}


def count_table(dev: pd.DataFrame, flag_count: pd.Series) -> list[dict]:
    rows = []
    for count, part in dev.groupby(flag_count, sort=True):
        y = part[TARGET].to_numpy()
        rows.append({
            "flag_count": int(count),
            "rows": int(len(part)),
            "frauds": int(y.sum()),
            "fraud_rate": float(y.mean()),
            "auc": safe_auc(y, part["p"].to_numpy()),
        })
    return rows


def leftover_columns(dev: pd.DataFrame, loud: pd.Series) -> list[dict]:
    names = [
        "amount_usd", "account_balance_usd", "distance_from_home_km", "hours_since_last_txn",
        "card_age_months", "customer_age", "time_of_day_hour", "day_of_week", "prior_disputes",
        "merchant_risk_score", "velocity_score", "cvv_retry_count", "txn_count_last_24h",
        "merchant_category", "card_type", "channel", "device_type",
    ]
    loud_frame = dev.loc[loud]
    y = loud_frame[TARGET]
    rows = []
    for name in names:
        column = loud_frame[name]
        if pd.api.types.is_numeric_dtype(column):
            auc = oriented_auc(y.to_numpy(), column.to_numpy(dtype=float))
            kind = "numeric_direction_free"
        else:
            # 用组内欺诈率，不用留一率。留一率会把同一类里的欺诈排到正常交易下面，弱列会被算成反向。
            auc = safe_auc(y.to_numpy(), group_rate(y, column).to_numpy())
            kind = "category_fraud_rate"
        rows.append({"column": name, "kind": kind, "loud_auc": auc})
    rows.sort(key=lambda row: row["loud_auc"] or 0, reverse=True)
    return rows


def decide(within_auc: float, quiet_overlap: bool, dead_columns: list[str]) -> dict:
    return {
        "action": "停止在现有列上继续加特征或更换模型",
        "reason": (
            f"同一组标记内部 AUC {within_auc:.4f} 是当前融合已经排出的成绩，主要来自商户风险、交易速度和 CVV 次数，不是待做的新步骤。"
            "夜间、距离、余额、金额、星期、卡组织和渠道在有标记交易上接近随机。"
            "没有这些标记的欺诈，分数落在同类正常交易内部。"
        ),
        "within_pattern_auc_is_current_model": True,
        "quiet_fraud_inside_normal_mass": quiet_overlap,
        "near_chance_columns": dead_columns,
    }


def markdown(result: dict) -> str:
    zones = result["zones"]
    decision = result["decision"]
    pieces = result["pair_decomposition"]["pieces"]
    lines = [
        "# 当前融合的排序上限",
        "",
        "范围：开发集折外预测。概率为 `xgb_r2_parts_smooth80` × 0.7 + `cat_r2_parts` × 0.3。不训练新模型，不使用已经看过的 20% 独立测试集，不改变当前提交。",
        "",
        f"结论：{decision['action']}。{decision['reason']}",
        "",
        "格子内的 0.84 不能读成“还有一块没做”。那是现有融合在标记相同的交易里，靠风险分、速度和 CVV 次数已经拉开的距离。",
        "",
        "## 全体分数从哪里来",
        "",
        f"开发集 {zones['all']['rows']} 行，欺诈 {zones['all']['frauds']} 笔，融合 AUC {zones['all']['auc']:.4f}。有任一响亮标记的交易 {zones['loud']['rows']} 行，其中欺诈 {zones['loud']['frauds']} 笔，内部 AUC {zones['loud']['auc']:.4f}。没有任何这些标记的交易 {zones['quiet']['rows']} 行，欺诈 {zones['quiet']['frauds']} 笔，内部 AUC {zones['quiet']['auc']:.4f}。",
        "",
        "把全部欺诈—正常配对拆开后，融合的全体 AUC 主要由“有标记的欺诈”对上“无标记的正常交易”贡献：",
        "",
        "| 配对 | 占全部配对 | 欺诈分更高的比例 | 对全体 AUC 的贡献 |",
        "| --- | ---: | ---: | ---: |",
    ]
    labels = {
        "loud_fraud_vs_quiet_normal": "有标记欺诈对无标记正常",
        "loud_fraud_vs_loud_normal": "有标记欺诈对有标记正常",
        "quiet_fraud_vs_quiet_normal": "无标记欺诈对无标记正常",
        "quiet_fraud_vs_loud_normal": "无标记欺诈对有标记正常",
    }
    for piece in pieces:
        lines.append(
            f"| {labels[piece['pair']]} | {piece['pair_share']:.1%} | {piece['win_rate']:.1%} | {piece['auc_contribution']:.4f} |"
        )
    lines.extend([
        "",
        f"四类配对加总为 {result['pair_decomposition']['reconstructed_auc']:.4f}，与直接计算的全体 AUC {zones['all']['auc']:.4f} 一致。有标记欺诈对上无标记正常交易，已经有 99.4% 排对。有标记交易内部的胜率是 92.4%，剩下的混排主要是高分正常交易和高分欺诈长得一样。没有标记的 540 笔欺诈只占全部配对的 3.3%，分数落在同类正常交易里面。",
        "",
        "## 同一组标记内部",
        "",
        f"12 个标记完全相同的格子里，融合仍能排出的加权 AUC 为 {result['within_exact_pattern']['weighted_auc']:.4f}。只按这 12 个标记的留一欺诈率排序，全体 AUC 为 {result['flag_pattern_auc']:.4f}；融合全体 AUC 比它高 {result['overall_gap_over_flags']:.4f}。在有标记交易内部，融合 AUC {zones['loud']['auc']:.4f}，标记组合 AUC {result['loud_flag_pattern_auc']:.4f}，差值 {result['loud_gap_over_flags']:.4f}。",
        "",
        "有标记欺诈里分数最低的 20%，和分数最高的 20%，以及有标记正常交易里分数最高的 1%：",
        "",
        "| 组别 | 行数 | 概率中位数 | 标记数中位数 | 风险分中位数 | 速度中位数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    group_labels = {
        "low_loud_fraud": "低分有标记欺诈",
        "high_loud_fraud": "高分有标记欺诈",
        "top_loud_normal": "高分有标记正常",
    }
    for key, label in group_labels.items():
        group = result["groups"][key]
        lines.append(
            f"| {label} | {group['rows']} | {group['score']['p50']:.4f} | {group['flag_count_median']:.0f} | {group['merchant_risk_score_median']:.1f} | {group['velocity_score_median']:.1f} |"
        )
    lines.extend([
        "",
        f"无标记欺诈的概率中位数是 {zones['quiet']['fraud_score']['p50']:.4f}，无标记正常交易的 90 分位是 {zones['quiet']['normal_score']['p90']:.4f}。这些欺诈的分数落在正常交易的分数范围里。",
        "",
        "## 其余列在有标记交易内部的单独区分度",
        "",
        "下表只在有标记交易上计算。数值列取正反两个方向中较高的 AUC；类别列用该列自己的欺诈率。这是单列还能不能分开，不是新模型的成绩。风险分、速度、CVV 次数和年龄模型里已经有了。",
        "",
        "| 列 | AUC |",
        "| --- | ---: |",
    ])
    for row in result["leftover_columns"]:
        lines.append(f"| {row['column']} | {row['loud_auc']:.4f} |")
    outside_auc = result["decision"]["merchant_category_auc_without_gift_crypto_game"]
    age = result["decision"]["loud_age_fraud_rate"]
    lines.extend([
        "",
        f"去掉礼品卡、加密货币和游戏之后，商户类别在剩余有标记交易上的欺诈率 AUC 为 {outside_auc:.4f}。有标记交易里，{age['youngest_age']} 岁的欺诈率是 {age['youngest_rate']:.2%}，{age['oldest_age']} 岁是 {age['oldest_rate']:.2%}。年龄已经作为数值列进入模型，欺诈率是随年龄抬高的，不是风险整数档那种一块一块跳的形状。",
    ])
    lines.extend([
        "",
        "## 标记个数",
        "",
        "| 标记数 | 行数 | 欺诈 | 欺诈率 | 组内 AUC |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in result["flag_counts"]:
        auc = "—" if row["auc"] is None else f"{row['auc']:.4f}"
        lines.append(f"| {row['flag_count']} | {row['rows']} | {row['frauds']} | {row['fraud_rate']:.2%} | {auc} |")
    lines.extend([
        "",
        "脚本：`scripts/ranking_ceiling.py`。原始数字：`reports/ranking_ceiling.json`。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    dev = build_frame()
    flags = flag_frame(dev)
    flags.index = dev.index
    loud = flags.any(axis=1)
    pattern = flags.astype(np.uint8).astype(str).agg("|".join, axis=1)
    flag_count = flags.sum(axis=1).astype(int)

    dev = dev.copy()
    dev["pattern"] = pattern
    dev["flag_count"] = flag_count
    pattern_rate = leave_one_out_rate(dev[TARGET], dev["pattern"])

    zones = {
        "all": zone_record(dev),
        "loud": zone_record(dev.loc[loud]),
        "quiet": zone_record(dev.loc[~loud]),
    }
    direct_auc = zones["all"]["auc"]
    parts = decomposition(dev, loud)
    if abs(parts["reconstructed_auc"] - direct_auc) > 1e-9:
        raise ValueError("配对分解与直接 AUC 不一致")

    fraud = dev.loc[loud & dev[TARGET].eq(1)]
    normal = dev.loc[loud & dev[TARGET].eq(0)]
    low_cut = float(fraud["p"].quantile(0.2))
    high_cut = float(fraud["p"].quantile(0.8))
    top_cut = float(normal["p"].quantile(0.99))
    groups = {
        "low_loud_fraud": group_profile(fraud.loc[fraud["p"].le(low_cut)], flags),
        "high_loud_fraud": group_profile(fraud.loc[fraud["p"].ge(high_cut)], flags),
        "top_loud_normal": group_profile(normal.loc[normal["p"].ge(top_cut)], flags),
    }
    groups["low_loud_fraud"]["score_max"] = low_cut
    groups["high_loud_fraud"]["score_min"] = high_cut
    groups["top_loud_normal"]["score_min"] = top_cut

    within = weighted_within_auc(dev, dev["pattern"])
    loud_flag_auc = safe_auc(dev.loc[loud, TARGET].to_numpy(), pattern_rate.loc[loud].to_numpy())
    overall_flag_auc = safe_auc(dev[TARGET].to_numpy(), pattern_rate.to_numpy())
    loud_gap = zones["loud"]["auc"] - loud_flag_auc
    overall_gap = direct_auc - overall_flag_auc

    quiet = zones["quiet"]
    quiet_overlap = quiet["fraud_score"]["p50"] <= quiet["normal_score"]["p90"]
    leftover = leftover_columns(dev, loud)
    outside = ~dev["merchant_category"].isin(["Gift Cards", "Crypto Exchange", "Gaming"])
    merchant_outside = safe_auc(
        dev.loc[loud & outside, TARGET].to_numpy(),
        group_rate(dev.loc[loud & outside, TARGET], dev.loc[loud & outside, "merchant_category"]).to_numpy(),
    )
    dead_columns = [row["column"] for row in leftover if row["loud_auc"] is not None and row["loud_auc"] < 0.56]
    age_table = dev.loc[loud].groupby("customer_age")[TARGET].agg(["size", "mean"])
    age_rate = age_table.loc[age_table["size"].ge(500), "mean"].sort_index()
    decision = decide(within["weighted_auc"], quiet_overlap, dead_columns)
    decision["merchant_category_auc_without_gift_crypto_game"] = merchant_outside
    decision["loud_age_fraud_rate"] = {
        "youngest_age": int(age_rate.index.min()),
        "youngest_rate": float(age_rate.iloc[0]),
        "oldest_age": int(age_rate.index.max()),
        "oldest_rate": float(age_rate.iloc[-1]),
    }

    result = {
        "created_at": timestamp(),
        "scope": "开发集折外预测。不训练，不读取独立测试集，不生成提交。",
        "blend": {"xgb": XGB_NAME, "cat": CAT_NAME, "xgb_weight": XGB_WEIGHT, "cat_weight": 0.3},
        "flags": list(flags.columns),
        "zones": zones,
        "pair_decomposition": parts,
        "within_exact_pattern": within,
        "flag_pattern_auc": overall_flag_auc,
        "loud_flag_pattern_auc": loud_flag_auc,
        "overall_gap_over_flags": overall_gap,
        "loud_gap_over_flags": loud_gap,
        "groups": groups,
        "flag_counts": count_table(dev, dev["flag_count"]),
        "leftover_columns": leftover,
        "decision": decision,
    }
    write_json(REPORT_JSON, result)
    REPORT_MD.write_text(markdown(result), encoding="utf-8")
    print(json.dumps({
        "all_auc": direct_auc,
        "loud_auc": zones["loud"]["auc"],
        "quiet_auc": zones["quiet"]["auc"],
        "within_pattern_auc": within["weighted_auc"],
        "loud_gap_over_flags": loud_gap,
        "action": decision["action"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
