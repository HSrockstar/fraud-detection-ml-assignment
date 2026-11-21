"""执行已获确认的固定 LLM 请求，并在相同样本上比较既有模型。"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from dotenv import dotenv_values
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                            confusion_matrix, f1_score, log_loss, precision_score,
                            recall_score, roc_auc_score)
from threadpoolctl import threadpool_limits

from .common import ID, ROOT, TARGET, timestamp, write_json


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol() -> tuple[Path, dict, list[dict]]:
    cfg_path = ROOT / "configs/llm_pilot.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    folder = ROOT / cfg["artifact_dir"]
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if sha256(cfg_path) != manifest["config_sha256"]:
        raise ValueError("实验配置已改变，禁止调用 API")
    prompt = (ROOT / cfg["prompt_path"]).read_text(encoding="utf-8")
    if hashlib.sha256(prompt.encode()).hexdigest() != manifest["prompt_sha256"]:
        raise ValueError("提示词已改变，禁止调用 API")
    query = pd.read_csv(folder / "无标签查询.csv")
    requests = []
    for batch in manifest["batches"]:
        path = folder / batch["file"]
        if sha256(path) != batch["sha256"]:
            raise ValueError(f"已确认的请求发生变化：{path.name}")
        request = json.loads(path.read_text(encoding="utf-8"))
        if request["model"] != "gpt-6.2-sol" or request["reasoning"] != {"effort": "xhigh"}:
            raise ValueError("模型与推理档位不符合已确认方案")
        if request["input"][0]["content"] != prompt or request["store"] is not False:
            raise ValueError("固定提示词或存储设置不符合已确认方案")
        block = request["input"][-1]["content"].split("<待预测交易>\n")[1].split("</待预测交易>")[0]
        frame = pd.read_csv(io.StringIO(block))
        if TARGET in frame or ID in frame or len(frame) != batch["rows"]:
            raise ValueError("查询数据包含标签、原始 id 或样本数不符")
        requests.append({"payload": request, "ids": frame["样本编号"].tolist(), **batch})
    all_ids = [sample for batch in requests for sample in batch["ids"]]
    if all_ids != query["样本编号"].tolist() or len(set(all_ids)) != 256:
        raise ValueError("请求未完整且唯一地覆盖已确认的 256 条查询")
    return folder, manifest, requests


def parse_predictions(response: dict, expected_ids: list[str]) -> list[dict]:
    actual_model = response.get("model", "")
    if actual_model != "gpt-6.2-sol" and not actual_model.startswith("gpt-6.2-sol-"):
        raise ValueError(f"API 返回了不同模型：{actual_model}")
    if response.get("status") != "completed":
        raise ValueError(f"API 响应未完成：{response.get('status')}，{response.get('incomplete_details')}")
    parts = []
    for output in response.get("output", []):
        if output.get("type") == "message":
            for content in output.get("content", []):
                if content.get("type") == "refusal":
                    raise ValueError("模型拒绝了本批预测")
                if content.get("type") == "output_text":
                    parts.append(content["text"])
    result = json.loads("".join(parts))
    if set(result) != {"预测结果"} or not isinstance(result["预测结果"], list):
        raise ValueError("预测结果结构不符")
    rows = result["预测结果"]
    if [row.get("样本编号") for row in rows] != expected_ids:
        raise ValueError("返回样本编号存在遗漏、重复、新增或顺序错误")
    for row in rows:
        if set(row) != {"样本编号", "欺诈概率"}:
            raise ValueError("预测行字段不符")
        probability = row["欺诈概率"]
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError("欺诈概率必须是数值")
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("欺诈概率必须有限且位于 [0,1]")
    return rows


def redact(message: str, key: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9_.*-]+", "[已隐藏密钥]", message.replace(key, "[已隐藏密钥]"))


def archive_truncated_attempt(response_dir: Path, number: int) -> dict:
    response_path = response_dir / f"response_{number:02d}.json"
    call_path = response_dir / f"call_{number:02d}.json"
    raw = json.loads(response_path.read_text(encoding="utf-8"))
    call = json.loads(call_path.read_text(encoding="utf-8"))
    if raw.get("status") != "incomplete" or raw.get("incomplete_details", {}).get("reason") != "max_output_tokens":
        raise ValueError("仅允许补跑已证实因输出 token 上限而截断的批次")
    archived_response = response_dir / f"incomplete_response_{number:02d}.json"
    archived_call = response_dir / f"incomplete_call_{number:02d}.json"
    if archived_response.exists() or archived_call.exists():
        raise ValueError("已保存一次失败尝试，禁止自动重复补跑")
    response_path.rename(archived_response)
    call_path.rename(archived_call)
    return {"retry_reason": "max_output_tokens", "previous_response_id": raw.get("id"),
            "previous_seconds": call["seconds"], "archived_response": archived_response.name}


def execute(limit_batches: int = 8, start_batch: int = 1, retry_token_limit: bool = False) -> None:
    if not 1 <= start_batch <= limit_batches <= 8:
        raise ValueError("批次范围必须位于 1 到 8，起点不得超过终点")
    if retry_token_limit and start_batch != limit_batches:
        raise ValueError("补跑只能指定一个已截断批次")
    folder, manifest, batches = load_protocol()
    values = dotenv_values(ROOT / ".env")
    key = (values.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise ValueError("项目根目录 .env 中缺少可用的 OPENAI_API_KEY")
    base_url = values.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    parsed = urlsplit(base_url)
    if parsed.scheme != "https" or parsed.hostname != "api.openai.com" or parsed.path.rstrip("/") != "/v1":
        raise ValueError("本次只向已确认的官方 https://api.openai.com/v1 发送请求")
    if values.get("OPENAI_MODEL") != "gpt-6.2-sol" or values.get("OPENAI_REASONING_EFFORT") != "xhigh":
        raise ValueError(".env 模型设置与已确认方案不一致")
    approval_path = folder / "approval.json"
    if not approval_path.exists():
        write_json(approval_path, {"recorded_at": timestamp(), "authorization": "用户已明确确认提示词、传数方案及 256 条样本预测",
                                   "prompt_sha256": manifest["prompt_sha256"], "request_hashes": [b["sha256"] for b in batches]})
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if approval["prompt_sha256"] != manifest["prompt_sha256"] or approval["request_hashes"] != [b["sha256"] for b in batches]:
        raise ValueError("确认记录与固定请求不一致")
    response_dir = folder / "responses"
    response_dir.mkdir(exist_ok=True)
    for number, batch in enumerate(batches[:limit_batches], 1):
        if number < start_batch:
            continue
        output_path = response_dir / f"response_{number:02d}.json"
        metadata_path = response_dir / f"call_{number:02d}.json"
        retry_details = archive_truncated_attempt(response_dir, number) if retry_token_limit else {}
        if output_path.exists():
            parse_predictions(json.loads(output_path.read_text(encoding="utf-8")), batch["ids"])
            print(f"第 {number}/8 批：复用已校验结果，不重复调用", flush=True)
            continue
        if metadata_path.exists():
            raise RuntimeError(f"第 {number} 批存在未完成调用记录；先检查错误，禁止自动重复计费")
        payload = dict(batch["payload"])
        if retry_token_limit:
            # 仅提高生成预算，已确认的消息、数据和模型参数逐字保持原样。
            payload["max_output_tokens"] = 32768
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        metadata = {"batch": number, "started_at": timestamp(), "request_sha256": batch["sha256"], "status": "calling",
                    "effective_max_output_tokens": payload["max_output_tokens"],
                    "effective_payload_sha256": hashlib.sha256(body).hexdigest(), **retry_details}
        write_json(metadata_path, metadata)
        print(f"第 {number}/8 批：正在预测 {len(batch['ids'])} 条样本", flush=True)
        request = Request(base_url.rstrip("/") + "/responses", data=body,
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
        start = time.perf_counter()
        try:
            with urlopen(request, timeout=900) as stream:
                raw = json.loads(stream.read().decode("utf-8"))
                metadata["request_id"] = stream.headers.get("x-request-id")
            # 即使解析失败也保留真实返回与用量，便于检查，不自动再次发起预测。
            write_json(output_path, raw)
            metadata.update({"seconds": time.perf_counter() - start, "response_id": raw.get("id"),
                             "actual_model": raw.get("model"), "usage": raw.get("usage"), "service_tier": raw.get("service_tier")})
            rows = parse_predictions(raw, batch["ids"])
            metadata.update({"status": "completed", "validated_rows": len(rows), "completed_at": timestamp()})
            write_json(metadata_path, metadata)
            print(f"第 {number}/8 批完成：{len(rows)} 条，{metadata['seconds']:.1f} 秒，用量 {raw.get('usage')}", flush=True)
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            metadata.update({"status": "http_error", "http_status": exc.code,
                             "seconds": time.perf_counter() - start, "error": redact(message, key)})
            write_json(metadata_path, metadata)
            raise RuntimeError(f"第 {number} 批 HTTP {exc.code}：{metadata['error']}") from None
        except (URLError, TimeoutError, OSError, ValueError, KeyError, TypeError) as exc:
            metadata.update({"status": "failed", "seconds": time.perf_counter() - start, "error": redact(str(exc), key)})
            write_json(metadata_path, metadata)
            raise RuntimeError(f"第 {number} 批失败：{metadata['error']}") from None


def weighted_metrics(y, p, weights, threshold=0.5) -> dict:
    prediction = p >= threshold
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "weighted_average_precision": float(average_precision_score(y, p, sample_weight=weights)),
        "weighted_log_loss": float(log_loss(y, p, sample_weight=weights, labels=[0, 1])),
        "weighted_brier": float(brier_score_loss(y, p, sample_weight=weights)),
        "weighted_accuracy": float(accuracy_score(y, prediction, sample_weight=weights)),
        "weighted_precision": float(precision_score(y, prediction, sample_weight=weights, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "weighted_f1": float(f1_score(y, prediction, sample_weight=weights, zero_division=0)),
        "threshold": float(threshold),
        "sample_confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
    }


def paired_bootstrap(y: np.ndarray, probabilities: dict[str, np.ndarray], seed=20261121, repeats=4000) -> dict:
    positives, negatives = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    pos_counts = rng.multinomial(len(positives), np.full(len(positives), 1 / len(positives)), size=repeats)
    neg_counts = rng.multinomial(len(negatives), np.full(len(negatives), 1 / len(negatives)), size=repeats)
    scores = {}
    with threadpool_limits(limits=6):
        for name, p in probabilities.items():
            pairwise = (p[positives, None] > p[None, negatives]).astype(float)
            pairwise += 0.5 * (p[positives, None] == p[None, negatives])
            scores[name] = ((pos_counts @ pairwise) * neg_counts).sum(axis=1) / (len(positives) * len(negatives))
    gap = scores["gpt-6.2-sol_xhigh"] - scores["selected"]
    return {"method": "按类别分层、同样本配对 bootstrap；只反映固定预测在抽样方面的不确定性",
            "repeats": repeats, "seed": seed,
            "auc_95_ci": {name: np.quantile(values, [0.025, 0.975]).tolist() for name, values in scores.items()},
            "gpt_minus_selected_auc_95_ci": np.quantile(gap, [0.025, 0.975]).tolist()}


def evaluate() -> None:
    folder, manifest, batches = load_protocol()
    rows, calls = [], []
    for number, batch in enumerate(batches, 1):
        raw = json.loads((folder / "responses" / f"response_{number:02d}.json").read_text(encoding="utf-8"))
        rows.extend(parse_predictions(raw, batch["ids"]))
        call = json.loads((folder / "responses" / f"call_{number:02d}.json").read_text(encoding="utf-8"))
        if call["status"] != "completed" or call["request_sha256"] != batch["sha256"]:
            raise ValueError("调用记录未完成或与固定请求不一致")
        calls.append(call)
    successful_calls = len(calls)
    for path in sorted((folder / "responses").glob("incomplete_call_*.json")):
        calls.append(json.loads(path.read_text(encoding="utf-8")))
    labels = pd.read_csv(folder / "本地评估标签_不发送.csv")
    prediction = pd.DataFrame(rows).rename(columns={"欺诈概率": "gpt-6.2-sol_xhigh"})
    frame = labels.merge(prediction, on="样本编号", validate="one_to_one")
    existing = pd.read_parquet(folder / "existing_predictions.parquet")
    frame = frame.merge(existing.drop(columns=TARGET), on=ID, validate="one_to_one")
    if len(frame) != manifest["evaluation_rows"] or frame.isna().any().any():
        raise ValueError("预测合并后缺失、重复或样本数不符")
    model_names = [c for c in existing if c not in {ID, TARGET}] + ["gpt-6.2-sol_xhigh"]
    y = frame[TARGET].to_numpy(dtype=int)
    weights = frame["总体指标样本权重"].to_numpy()
    probabilities = {name: frame[name].to_numpy(dtype=float) for name in model_names}
    if any(not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() for p in probabilities.values()):
        raise ValueError("比较数据存在无效概率")
    results = {name: weighted_metrics(y, p, weights) for name, p in probabilities.items()}
    bootstrap = paired_bootstrap(y, probabilities)
    operating_points = {}
    selected_threshold = json.loads((ROOT / "reports/selection.json").read_text(encoding="utf-8"))["f1_threshold"]
    for name, p in probabilities.items():
        if name == "gpt-6.2-sol_xhigh":
            threshold = 0.5
        elif name == "selected":
            threshold = selected_threshold
        else:
            record = json.loads((ROOT / "reports/experiments" / f"{name}.json").read_text(encoding="utf-8"))
            threshold = record["oof_selected_threshold"]["threshold"]
        operating_points[name] = weighted_metrics(y, p, weights, threshold)
    usage = {name: sum(c["usage"].get(name, 0) for c in calls) for name in ["input_tokens", "output_tokens", "total_tokens"]}
    usage["cached_input_tokens"] = sum(c["usage"].get("input_tokens_details", {}).get("cached_tokens", 0) for c in calls)
    usage["cache_write_tokens"] = sum(c["usage"].get("input_tokens_details", {}).get("cache_write_tokens", 0) for c in calls)
    usage["cache_write_12h_tokens"] = sum(c["usage"].get("input_tokens_details", {}).get("cache_write_tokens_details", {}).get("ttl_12h_tokens", 0) for c in calls)
    usage["reasoning_tokens"] = sum(c["usage"].get("output_tokens_details", {}).get("reasoning_tokens", 0) for c in calls)
    usage["api_seconds"] = sum(c["seconds"] for c in calls)
    starts = [datetime.fromisoformat(c["started_at"]) for c in calls]
    ends = [start + timedelta(seconds=c["seconds"]) for start, c in zip(starts, calls)]
    usage["wall_seconds"] = (max(ends) - min(starts)).total_seconds()
    usage.update({"api_calls": len(calls), "completed_prediction_batches": successful_calls, "incomplete_calls": len(calls) - successful_calls})
    # 按 API 返回的缓存读取、写入和普通输入分别计价；最终金额以账单为准。
    uncached = usage["input_tokens"] - usage["cached_input_tokens"] - usage["cache_write_tokens"]
    write_30m = usage["cache_write_tokens"] - usage["cache_write_12h_tokens"]
    usage["estimated_standard_usd"] = (2 * uncached + 0.1 * usage["cached_input_tokens"] + 2.5 * write_30m
                                       + 5 * usage["cache_write_12h_tokens"] + 10 * usage["output_tokens"]) / 1e6
    output = ROOT / "reports/llm_pilot_v1"
    output.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame([{"model": name, **{k: v for k, v in score.items() if k != "sample_confusion_matrix"},
                                "auc_ci_lower": bootstrap["auc_95_ci"][name][0], "auc_ci_upper": bootstrap["auc_95_ci"][name][1]}
                               for name, score in results.items()]).sort_values("roc_auc", ascending=False)
    comparison.to_csv(output / "comparison.csv", index=False)
    # 保留 float32 模型概率提升为 float64 后的精度，便于 CSV 独立复算。
    frame.to_csv(output / "sample_predictions.csv", index=False, float_format="%.17g")
    report = {"completed_at": timestamp(), "model": "gpt-6.2-sol", "reasoning_effort": "xhigh",
              "training_rows": 512, "evaluation_rows": len(frame), "evaluation_fraud": int(y.sum()),
              "evaluation_normal": int(len(y) - y.sum()), "population_fraud_prior": float(np.average(y, weights=weights)),
              "metrics_at_common_threshold_0_5": results, "frozen_operating_points": operating_points,
              "bootstrap": bootstrap, "usage": usage,
              "gpt_minus_selected_auc": results["gpt-6.2-sol_xhigh"]["roc_auc"] - results["selected"]["roc_auc"],
              "scope": "既有留出集富集小样本；AUC 同批比较，AP/损失/精确率/准确率按原留出集类别比例加权；没有利用本批标签调整提示词或阈值。",
              "baseline_source": "artifacts/evaluation/holdout_predictions.parquet；模型仅在开发集训练，禁止使用全量重训模型。",
              "prompt_sha256": manifest["prompt_sha256"], "request_hashes": [b["sha256"] for b in batches]}
    write_json(output / "metrics.json", report)
    write_json(folder / "run_status.json", {"status": "completed", "api_calls": len(calls), "completed_batches": successful_calls,
                                           "incomplete_calls": len(calls) - successful_calls, "validated_rows": len(frame), "completed_at": timestamp()})
    print(comparison[["model", "roc_auc", "weighted_average_precision", "weighted_log_loss"]].to_string(index=False), flush=True)
    print(json.dumps({"usage": usage, "gpt_minus_selected_auc": report["gpt_minus_selected_auc"],
                      "paired_gap_95_ci": bootstrap["gpt_minus_selected_auc_95_ci"], "output": str(output)}, ensure_ascii=False), flush=True)
