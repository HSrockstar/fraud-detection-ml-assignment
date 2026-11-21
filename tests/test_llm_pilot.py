import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from fraud.llm_pilot import archive_truncated_attempt, execute, paired_bootstrap, parse_predictions, redact


def response(rows):
    return {"model": "gpt-6.2-sol", "status": "completed", "output": [
        {"type": "reasoning"},
        {"type": "message", "content": [{"type": "output_text", "text": json.dumps({"预测结果": rows})}]},
    ]}


def test_prediction_alignment_rejects_omitted_duplicate_and_reordered_ids():
    for ids in [["Q1"], ["Q1", "Q1"], ["Q2", "Q1"]]:
        with pytest.raises(ValueError, match="样本编号"):
            parse_predictions(response([{"样本编号": sample, "欺诈概率": 0.2} for sample in ids]), ["Q1", "Q2"])


@pytest.mark.parametrize("value", [True, float("nan"), -0.01, 1.01, "0.5"])
def test_invalid_probability_is_not_silently_accepted(value):
    with pytest.raises(ValueError, match="欺诈概率"):
        parse_predictions(response([{"样本编号": "Q1", "欺诈概率": value}]), ["Q1"])


def test_reasoning_item_does_not_hide_json_output():
    rows = [{"样本编号": "Q1", "欺诈概率": 0.02}]
    assert parse_predictions(response(rows), ["Q1"]) == rows


def test_unfinished_response_is_not_scored():
    raw = response([])
    raw["status"] = "incomplete"
    with pytest.raises(ValueError, match="未完成"):
        parse_predictions(raw, [])


def test_redaction_removes_complete_and_masked_credentials():
    assert "sk-" not in redact("bad key sk-dummy**** and secret", "secret")


def test_paired_bootstrap_preserves_identical_model_gap_with_ties():
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([0.1, 0.6, 0.6, 0.6, 0.1, 0.9])
    assert roc_auc_score(y, p) == pytest.approx(8 / 9)
    result = paired_bootstrap(y, {"selected": p, "gpt-6.2-sol_xhigh": p}, repeats=100)
    assert result["gpt_minus_selected_auc_95_ci"] == [0, 0]


def test_truncated_attempt_is_preserved_before_retry(tmp_path):
    raw = {"id": "response-incomplete", "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
    (tmp_path / "response_07.json").write_text(json.dumps(raw))
    (tmp_path / "call_07.json").write_text(json.dumps({"seconds": 10, "usage": {"output_tokens": 16384}}))
    details = archive_truncated_attempt(tmp_path, 7)
    assert details["previous_response_id"] == "response-incomplete"
    assert json.loads((tmp_path / "incomplete_response_07.json").read_text()) == raw
    assert json.loads((tmp_path / "incomplete_call_07.json").read_text())["usage"]["output_tokens"] == 16384
    assert not (tmp_path / "response_07.json").exists()


def test_completed_batch_cannot_be_retried_as_token_limit_failure(tmp_path):
    (tmp_path / "response_01.json").write_text(json.dumps({"status": "completed"}))
    (tmp_path / "call_01.json").write_text(json.dumps({"seconds": 10}))
    with pytest.raises(ValueError, match="仅允许"):
        archive_truncated_attempt(tmp_path, 1)
    assert (tmp_path / "response_01.json").exists()


def test_invalid_batch_range_is_rejected_before_loading_credentials():
    with pytest.raises(ValueError, match="批次范围"):
        execute(limit_batches=3, start_batch=4)
    with pytest.raises(ValueError, match="一个"):
        execute(limit_batches=8, start_batch=7, retry_token_limit=True)
