import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from fraud import common
from fraud.evaluation import f1_threshold, metrics
from fraud.features import cat_frame, engineer, preprocessor, value_keys
from fraud.models import _glm_margin, _train_glm, fit_model


def test_preprocessing_learns_only_training_data():
    train = pd.DataFrame({"amount": [1., 2., np.nan, 3.], "category": ["A", "B", "A", "A"]})
    valid = pd.DataFrame({"amount": [1e9, np.nan], "category": ["UNSEEN", "A"]})
    prep = preprocessor(train, scale=True).fit(train)
    before = prep.named_transformers_["numeric"].named_steps["impute"].statistics_.copy()
    encoded = prep.transform(valid)
    assert encoded.shape == (2, 3)
    assert np.isfinite(encoded).all()
    assert np.array_equal(before, [2.])
    assert np.array_equal(prep.named_transformers_["numeric"].named_steps["impute"].statistics_, before)
    assert "UNSEEN" not in prep.named_transformers_["categorical"].named_steps["encode"].categories_[0]


def test_feature_engineering_handles_zero_and_boolean_signals():
    x = pd.DataFrame({"amount_usd": [10., 20.], "account_balance_usd": [0., 100.],
                      "used_vpn": [True, False], "ip_country_mismatch": [True, True], "time_of_day_hour": [0, 6]})
    output = engineer(x, enabled=True)
    assert np.isnan(output.loc[0, "amount_balance_ratio"])
    assert output.loc[1, "amount_balance_ratio"] == pytest.approx(0.2)
    assert output.risk_flag_count.tolist() == [2, 1]
    assert output.night_transaction.tolist() == [1, 0]
    assert np.isfinite(output.select_dtypes("number").to_numpy()[~np.isnan(output.select_dtypes("number").to_numpy())]).all()
    assert "risk_flag_count" not in x


def test_threshold_optimizes_f1_on_supplied_validation_predictions():
    y = np.array([0, 0, 1, 0, 1, 1])
    p = np.array([0.01, 0.05, 0.2, 0.25, 0.3, 0.4])
    threshold = f1_threshold(y, p)
    assert f1_score(y, p >= threshold) == max(f1_score(y, p >= t) for t in p)


def _value_frame(amount):
    rows = len(amount)
    return pd.DataFrame({
        "amount_usd": amount,
        "account_balance_usd": np.full(rows, 10.0),
        "distance_from_home_km": np.full(rows, 1.0),
        "hours_since_last_txn": np.full(rows, 1.25),
        "velocity_score": np.full(rows, 5.5),
        "merchant_risk_score": np.full(rows, 29.2),
        "cvv_retry_count": np.zeros(rows, dtype=int),
        "auth_method": ["OTP"] * rows,
        "is_foreign_transaction": [False] * rows,
        "merchant_category": ["Groceries"] * rows,
        "channel": ["Web"] * rows,
        "device_type": ["Mobile"] * rows,
        "used_vpn": [False] * rows,
    })


def test_value_keys_are_label_free_tokens_and_do_not_mutate_input():
    frame = _value_frame([61.16, 0.0])
    encoded = value_keys(frame, "exact+parts+digits+interactions")
    assert frame.columns.tolist() == _value_frame([61.16, 0.0]).columns.tolist()
    assert encoded.loc[0, "vk_exact_amount_usd"] == "61.1600"
    assert encoded.loc[0, "vk_integer_amount_usd"] == "61"
    assert encoded.loc[0, "vk_fraction_amount_usd"] == "0.1600"
    assert [encoded.loc[0, f"vk_digit{n}_amount_usd"] for n in (1, 2, 3)] == ["6", "61", "611"]
    assert encoded.loc[1, "vk_digit1_amount_usd"] == "0"
    assert encoded.loc[0, "vk_cvv_auth"] == "0|OTP"


def test_value_key_columns_are_target_encoded_without_one_hot_or_label_leak():
    frame = _value_frame(np.arange(80) + 0.16)
    keyed = value_keys(frame, "exact")
    y = np.tile([0, 1], 40)
    prep = preprocessor(keyed, target_encode=True, seed=0)
    encoded = prep.fit_transform(keyed, y)
    names = prep.get_feature_names_out().tolist()
    column = names.index("target__vk_exact_amount_usd")
    assert encoded.shape[1] < len(frame)
    assert not any(name.startswith("categorical__vk_") for name in names)
    assert np.allclose(encoded[:, column], 0.5)
    unseen = keyed.iloc[[0]].copy()
    unseen["vk_exact_amount_usd"] = "99999.0000"
    assert prep.transform(unseen)[0, column] == pytest.approx(0.5)


def test_catboost_receives_part_bins_as_categories():
    frame = value_keys(_value_frame([61.16]), "parts")
    _encoded, columns = cat_frame(frame)
    assert "vk_integer_amount_usd" in columns
    assert "vk_fraction_amount_usd" in columns
    assert "amount_usd" not in columns


def test_score_pair_uses_integer_bins_not_exact_values():
    frame = _value_frame([10.0, 11.0])
    frame["merchant_risk_score"] = [77.4, 8.1]
    frame["velocity_score"] = [53.2, 0.7]
    encoded = value_keys(frame, "score_pair")
    assert encoded.loc[0, "vk_integer_merchant_risk_score"] == "77"
    assert encoded.loc[0, "vk_merchant_velocity_integer"] == "77|53"
    assert "vk_integer_amount_usd" not in encoded.columns


def test_tail_keys_keep_two_decimals_and_tens_groups():
    frame = _value_frame([1234.56, 61.16])
    encoded = value_keys(frame, "tails")
    assert encoded.loc[0, "vk_tail2_amount_usd"] == "56"
    assert encoded.loc[0, "vk_tens_amount_usd"] == "123"
    assert encoded.loc[1, "vk_tail2_amount_usd"] == "16"
    assert encoded.loc[1, "vk_tens_amount_usd"] == "6"
    assert "vk_integer_amount_usd" not in encoded.columns


def test_multiple_target_smooth_values_are_separate_columns():
    frame = pd.DataFrame({"amount": np.arange(40), "category": ["A", "B"] * 20})
    y = np.tile([0, 1], 20)
    prep = preprocessor(frame, target_encode=True, seed=0, target_smooth=[5, 80])
    encoded = prep.fit_transform(frame, y)
    names = prep.get_feature_names_out().tolist()
    assert encoded.shape[1] == len(names)
    assert "target_5__amount" in names and "target_80__amount" in names
    assert "target_5__category" in names and "target_80__category" in names
    assert not np.allclose(encoded[:, names.index("target_5__category")], encoded[:, names.index("target_80__category")])


def test_glm_margin_uses_training_rows_only():
    encoded = np.column_stack([np.arange(40), np.zeros(40)])
    y = np.array([0, 1] * 20)
    glm, scaler = _train_glm(encoded[:30], y[:30], 1.0)
    train_margin = _glm_margin(glm, scaler, encoded[:30])
    unseen_margin = _glm_margin(glm, scaler, encoded[30:])
    assert train_margin.shape == (30,)
    assert np.isfinite(train_margin).all() and np.isfinite(unseen_margin).all()
    assert train_margin[y[:30] == 1].mean() > train_margin[y[:30] == 0].mean()


def test_glm_boost_rejects_non_xgboost():
    frame = pd.DataFrame({"amount": [1.0, 2.0, 3.0, 4.0], "category": ["A", "B", "A", "B"]})
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="XGBoost"):
        fit_model("tree", {"glm_boost": True}, frame, y, seed=0)


def test_unknown_value_encoding_is_rejected():
    with pytest.raises(ValueError, match="未知取值编码"):
        value_keys(_value_frame([1.0]), "exact+exact")


def test_target_encoding_unique_training_values_do_not_reveal_own_labels():
    x = pd.DataFrame({"unique_value": np.arange(100)})
    y = np.tile([0, 1], 50)
    prep = preprocessor(x, target_encode=True, seed=42)
    encoded = prep.fit_transform(x, y)
    names = prep.get_feature_names_out().tolist()
    target_column = names.index("target__unique_value")
    # 唯一值在其他内部训练折中不存在，只能回退到 0.5，而不是自身标签。
    assert np.allclose(encoded[:, target_column], 0.5)
    assert prep.transform(pd.DataFrame({"unique_value": [10000]}))[0, target_column] == pytest.approx(0.5)


@pytest.mark.parametrize("probability", [[0.1, np.nan], [-0.1, 0.9], [0.1, 1.1]])
def test_invalid_probability_is_rejected(probability):
    with pytest.raises(ValueError, match="预测概率"):
        metrics([0, 1], probability)


def test_split_is_disjoint_stratified_persistent_and_config_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    configuration = {"seed": 42, "holdout_fraction": 0.2, "n_splits": 3}
    (tmp_path / "configs/experiment.json").write_text(json.dumps(configuration), encoding="utf-8")
    frame = pd.DataFrame({"id": np.arange(300), "is_fraud": np.tile([0, 0, 0, 0, 1], 60)})
    split = common.splits(frame)
    dev = split.partition == "dev"
    assert len(split) == 300 and split.id.is_unique
    assert (split.loc[~dev, "fold"] == -1).all()
    assert set(split.loc[dev, "fold"]) == {0, 1, 2}
    assert set(split.loc[dev, "id"]).isdisjoint(split.loc[~dev, "id"])
    for fold in range(3):
        assert frame.loc[split.fold == fold, "is_fraud"].mean() == pytest.approx(0.2)
    pd.testing.assert_frame_equal(split, common.splits(frame))
    configuration["seed"] = 43
    (tmp_path / "configs/experiment.json").write_text(json.dumps(configuration), encoding="utf-8")
    with pytest.raises(ValueError, match="数据或划分配置变化"):
        common.splits(frame)
