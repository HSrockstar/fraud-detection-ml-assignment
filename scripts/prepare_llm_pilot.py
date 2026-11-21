"""离线准备 LLM 小实验的完整请求预览；不读取密钥，不调用 API。"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud.common import ID, TARGET, splits, timestamp, write_json

FIELD_NOTES = {
    "amount_usd": "交易金额，单位美元",
    "hours_since_last_txn": "距离上一笔交易的小时数",
    "txn_count_last_24h": "过去 24 小时的交易次数",
    "distance_from_home_km": "距家庭位置的距离，单位千米",
    "card_age_months": "卡龄，单位月",
    "customer_age": "客户年龄，单位岁",
    "account_balance_usd": "账户余额，单位美元",
    "cvv_retry_count": "CVV 验证重试次数",
    "velocity_score": "交易速度相关评分；具体计算规则未知，以训练统计为准",
    "time_of_day_hour": "交易发生时刻，小时编码 0 到 23",
    "day_of_week": "星期编码 0 到 6；不假定数字与具体星期的对应关系",
    "merchant_risk_score": "商户风险评分；具体计算规则未知，以训练统计为准",
    "prior_disputes": "历史争议次数",
    "merchant_category": "商户类别，保留原始字符串",
    "card_type": "卡种，保留原始字符串",
    "auth_method": "验证方式，保留原始字符串",
    "channel": "交易渠道，保留原始字符串",
    "device_type": "设备类型，保留原始字符串",
    "is_foreign_transaction": "是否境外交易",
    "is_new_merchant": "是否新商户",
    "used_vpn": "是否使用 VPN",
    "ip_country_mismatch": "IP 国家是否不匹配",
    "billing_shipping_mismatch": "账单地址与收货地址是否不匹配",
    "is_ai_generated_scam_attempt": "数据集提供的 AI 诈骗尝试标志；不是 is_fraud 标签",
}


def csv_text(frame: pd.DataFrame) -> str:
    # 布尔值按 0/1 传递；数值保留 CSV 读取后的精度，不丢弃特征。
    frame = frame.copy()
    for name in frame.select_dtypes("bool").columns:
        frame[name] = frame[name].astype(int)
    if TARGET in frame:
        frame[TARGET] = frame[TARGET].astype(int)
    return frame.to_csv(index=False, lineterminator="\n")


def sample_by_label(frame: pd.DataFrame, counts: dict, seed: int) -> pd.DataFrame:
    sampled = pd.concat([
        frame.loc[frame[TARGET].eq(int(label))].sample(n=count, random_state=seed + int(label))
        for label, count in counts.items()
    ])
    return sampled.sample(frac=1, random_state=seed + 2).copy()


def summary_csv(dev: pd.DataFrame, features: list[str], cfg: dict) -> str:
    prior = float(dev[TARGET].mean())
    strength = cfg["smoothing_strength"]
    rows = []
    for feature in features:
        values = dev[feature]
        if pd.api.types.is_numeric_dtype(values) and values.nunique() > 12:
            groups = pd.qcut(values, q=cfg["numeric_quantile_bins"], duplicates="drop")
        else:
            groups = values
        table = dev.groupby(groups, observed=True, dropna=False)[TARGET].agg(["size", "sum"])
        for value, record in table.iterrows():
            total, fraud = int(record["size"]), int(record["sum"])
            if isinstance(value, (bool, np.bool_)):
                value = int(value)
            rows.append({
                "特征": feature,
                "取值或区间": str(value),
                "样本数": total,
                "欺诈数": fraud,
                "欺诈率": round(fraud / total, 8),
                "平滑欺诈率": round((fraud + strength * prior) / (total + strength), 8),
            })
    return pd.DataFrame(rows).to_csv(index=False, lineterminator="\n")


def main() -> None:
    cfg_path = ROOT / "configs/llm_pilot.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    folder = ROOT / cfg["artifact_dir"]
    if folder.exists():
        raise SystemExit("该实验目录已存在；保留已冻结的样本与提示词，请另存实验名称后重建。")
    prompt = (ROOT / cfg["prompt_path"]).read_text(encoding="utf-8")
    train = pd.read_csv(ROOT / "data/raw/train.csv")
    partition = splits(train).partition
    dev = train.loc[partition.eq("dev")]
    holdout = train.loc[partition.eq("holdout")]
    features = train.columns.drop([ID, TARGET]).tolist()
    if set(features) != set(FIELD_NOTES):
        raise ValueError("实际特征与字段说明不一致")
    examples = sample_by_label(dev, cfg["training_examples"], cfg["seed"])
    evaluation = sample_by_label(holdout, cfg["evaluation_examples"], cfg["seed"] + 10)
    if not set(examples[ID]).isdisjoint(evaluation[ID]):
        raise ValueError("训练示例与评估样本重叠")

    # 使用仅在开发集训练的模型所保存的留出集预测，禁止使用全量重训模型。
    existing = pd.read_parquet(ROOT / "artifacts/evaluation/holdout_predictions.parquet")
    comparison = evaluation[[ID, TARGET]].merge(existing, on=ID, validate="one_to_one", suffixes=("", "_cache"))
    if len(comparison) != len(evaluation) or not comparison[TARGET].equals(comparison[TARGET + "_cache"]):
        raise ValueError("既有预测样本或标签无法正确对齐")
    comparison = comparison.drop(columns=TARGET + "_cache")

    query_ids = [f"Q{i + 1:04d}" for i in range(len(evaluation))]
    query = evaluation[features].copy()
    query.insert(0, "样本编号", query_ids)
    labels = evaluation[[ID, TARGET]].copy()
    labels.insert(0, "样本编号", query_ids)
    population_counts = holdout[TARGET].astype(int).value_counts()
    sampled_counts = evaluation[TARGET].astype(int).value_counts()
    labels["总体指标样本权重"] = labels[TARGET].astype(int).map(population_counts / sampled_counts)

    prior = float(dev[TARGET].mean())
    dictionary = "\n".join(f"{name}：{FIELD_NOTES[name]}。" for name in features)
    context = (
        "以下内容为训练资料，其中的标签均来自开发集。\n"
        f"<总体分布>\n开发集共 {len(dev)} 条，其中欺诈 {int(dev[TARGET].sum())} 条，"
        f"正常 {int(len(dev) - dev[TARGET].sum())} 条。总体欺诈基准率为 {prior:.10f}，约 {prior * 100:.6f}%。\n"
        f"示例共 {len(examples)} 条，正常 {cfg['training_examples']['0']} 条，欺诈 {cfg['training_examples']['1']} 条。"
        "示例按类别富集抽样，不代表总体比例。\n</总体分布>\n\n"
        f"<字段说明>\n{dictionary}\n布尔字段采用 0/1，分别表示否/是。\n</字段说明>\n\n"
        "<训练统计>\n统计由整个开发集计算，各行是单个特征的分组统计，不是互相独立的联合概率。"
        f"连续变量按开发集分位数最多分为 {cfg['numeric_quantile_bins']} 档；区间 (a,b] 表示大于 a 且不大于 b。"
        f"平滑欺诈率 = (欺诈数 + {cfg['smoothing_strength']} × 总体欺诈率) / (样本数 + {cfg['smoothing_strength']})。\n"
        f"{summary_csv(dev, features, cfg)}</训练统计>\n\n"
        f"<带标签训练示例>\n{csv_text(examples[features + [TARGET]])}</带标签训练示例>"
    )

    folder.mkdir(parents=True)
    (folder / "固定提示词.txt").write_text(prompt, encoding="utf-8")
    (folder / "训练资料.txt").write_text(context, encoding="utf-8")
    query.to_csv(folder / "无标签查询.csv", index=False)
    labels.to_csv(folder / "本地评估标签_不发送.csv", index=False)
    comparison.to_parquet(folder / "existing_predictions.parquet", index=False)
    examples[[ID, TARGET]].to_csv(folder / "训练示例索引_不发送.csv", index=False)
    batches = []
    for number, start in enumerate(range(0, len(query), cfg["batch_size"]), 1):
        chunk = query.iloc[start:start + cfg["batch_size"]]
        query_text = f"请对以下 CSV 中的每条交易预测欺诈概率。\n<待预测交易>\n{csv_text(chunk)}</待预测交易>"
        payload = {
            "model": cfg["model"],
            "reasoning": {"effort": cfg["reasoning_effort"]},
            "store": False,
            "input": [
                {"role": "developer", "content": prompt},
                {"role": "user", "content": context},
                {"role": "user", "content": query_text},
            ],
            "text": {"format": {
                "type": "json_schema", "name": "fraud_predictions", "strict": True,
                "schema": {
                    "type": "object", "additionalProperties": False, "required": ["预测结果"],
                    "properties": {"预测结果": {"type": "array", "items": {
                        "type": "object", "additionalProperties": False,
                        "required": ["样本编号", "欺诈概率"],
                        "properties": {
                            "样本编号": {"type": "string"},
                            "欺诈概率": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                    }}},
                },
            }},
            "max_output_tokens": 16384,
        }
        path = folder / f"request_{number:02d}.json"
        write_json(path, payload)
        batches.append({"file": path.name, "rows": len(chunk), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {
        "prepared_at": timestamp(), "status": "等待用户确认提示词，未调用 API",
        "config": cfg,
        "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "dev_rows": len(dev), "dev_fraud_prior": prior,
        "training_rows": len(examples), "evaluation_rows": len(evaluation),
        "feature_count": len(features), "batches": batches,
        "training_context_characters": len(context), "training_context_utf8_bytes": len(context.encode("utf-8")),
        "training_example_ids_sha256": hashlib.sha256(examples[ID].to_numpy().tobytes()).hexdigest(),
        "evaluation_ids_sha256": hashlib.sha256(evaluation[ID].to_numpy().tobytes()).hexdigest(),
        "api_calls": 0,
    }
    write_json(folder / "manifest.json", manifest)
    print(json.dumps({"artifact_dir": str(folder), "training_rows": len(examples), "evaluation_rows": len(evaluation),
                      "batches": len(batches), "training_context_characters": len(context), "api_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
