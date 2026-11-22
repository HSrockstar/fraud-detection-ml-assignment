"""无标签特征构造；所有学习型预处理均在模型训练数据上拟合。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder


def engineer(frame: pd.DataFrame, enabled: bool = False) -> pd.DataFrame:
    result = frame.copy()
    if not enabled:
        return result
    # 仅使用本行可获得的信息，分母为零时交给训练折内的缺失值处理。
    def ratio(name, numerator, denominator):
        if numerator in result and denominator in result:
            result[name] = result[numerator] / result[denominator].replace(0, np.nan)
    ratio("amount_balance_ratio", "amount_usd", "account_balance_usd")
    if "amount_usd" in result:
        result["log_amount_usd"] = np.log1p(result["amount_usd"].clip(lower=0))
    if "distance_from_home_km" in result:
        result["log_distance"] = np.log1p(result["distance_from_home_km"].clip(lower=0))
    if "time_of_day_hour" in result:
        angle = 2 * np.pi * result["time_of_day_hour"] / 24
        result["hour_sin"], result["hour_cos"] = np.sin(angle), np.cos(angle)
        result["night_transaction"] = (result["time_of_day_hour"] < 6).astype("int8")
    flags = [c for c in ["used_vpn", "ip_country_mismatch", "is_foreign_transaction", "billing_shipping_mismatch", "is_new_merchant"] if c in result]
    if flags:
        result["risk_flag_count"] = result[flags].sum(axis=1)
    for a, b, name in [
        ("merchant_risk_score", "velocity_score", "merchant_velocity_risk"),
        ("merchant_risk_score", "amount_usd", "merchant_amount_risk"),
        ("distance_from_home_km", "velocity_score", "distance_velocity"),
    ]:
        if a in result and b in result:
            result[name] = result[a] * result[b]
    return result.replace([np.inf, -np.inf], np.nan)


KEY_PREFIX = "vk_"
VALUE_KEY_COLUMNS = (
    "amount_usd", "account_balance_usd", "distance_from_home_km",
    "hours_since_last_txn", "velocity_score", "merchant_risk_score",
)
VALUE_KEY_MODES = ("exact", "parts", "digits", "tails", "interactions", "interactions_ext", "score_integers", "score_pair")
SCORE_INTEGER_COLUMNS = ("merchant_risk_score", "velocity_score")


def _numeric_values(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")


def _string_tokens(index: pd.Index, values: np.ndarray, labels: np.ndarray) -> pd.Series:
    output = pd.Series(pd.NA, index=index, dtype="string")
    mask = np.isfinite(values)
    output.loc[mask] = labels[mask]
    return output


def _exact_tokens(series: pd.Series) -> pd.Series:
    values = _numeric_values(series)
    labels = np.char.mod("%.4f", np.round(np.nan_to_num(values), 4))
    return _string_tokens(series.index, values, labels)


def _part_tokens(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    values = _numeric_values(series)
    finite = np.nan_to_num(values)
    integer = np.char.mod("%.0f", np.trunc(finite))
    fraction = np.char.mod("%.4f", np.round(finite - np.trunc(finite), 4))
    return _string_tokens(series.index, values, integer), _string_tokens(series.index, values, fraction)


def _tail_tokens(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    values = _numeric_values(series)
    finite = np.nan_to_num(values)
    fraction = np.abs(finite - np.trunc(finite))
    tail = np.clip(np.rint(fraction * 100), 0, 99)
    tens = np.floor(np.trunc(finite) / 10.0)
    return (
        _string_tokens(series.index, values, np.char.mod("%.0f", tail)),
        _string_tokens(series.index, values, np.char.mod("%.0f", tens)),
    )


def _digit_tokens(series: pd.Series, digits: int) -> pd.Series:
    values = _numeric_values(series)
    labels = np.empty(len(values), dtype=object)
    labels[:] = None
    zeros = np.isfinite(values) & (values == 0)
    positive = np.isfinite(values) & (np.abs(values) > 0)
    labels[zeros] = "0"
    magnitude = np.abs(values[positive])
    exponent = np.floor(np.log10(magnitude))
    scaled = np.floor(magnitude / np.power(10.0, exponent - (digits - 1)) + 1e-8)
    labels[np.flatnonzero(positive)] = scaled.astype(np.int64).astype(str)
    return _string_tokens(series.index, values, labels)


def _category_token(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("int8").astype("string")
    if pd.api.types.is_integer_dtype(series):
        return series.astype("string")
    if pd.api.types.is_numeric_dtype(series):
        return _exact_tokens(series)
    return series.astype("string")


def _interaction(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> pd.Series:
    parts = [_category_token(frame[column]) for column in columns]
    return parts[0].str.cat(parts[1:], sep="|", na_rep="__MISSING__")


def value_keys(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    """由原始输入构造类别键。键本身不含标签，目标编码仍在训练折内拟合。"""
    selected = tuple(mode.split("+"))
    unknown = [item for item in selected if item not in VALUE_KEY_MODES]
    if unknown or len(selected) != len(set(selected)):
        raise ValueError(f"未知取值编码: {mode}")
    result = frame.copy()
    needed = []
    if {"exact", "parts", "digits", "tails"} & set(selected):
        needed.extend(VALUE_KEY_COLUMNS)
    if {"score_integers", "score_pair"} & set(selected):
        needed.extend(SCORE_INTEGER_COLUMNS)
    missing = [column for column in dict.fromkeys(needed) if column not in result.columns]
    if missing:
        raise ValueError(f"缺少取值编码列: {missing}")
    for column in VALUE_KEY_COLUMNS:
        if "exact" in selected:
            result[f"{KEY_PREFIX}exact_{column}"] = _exact_tokens(result[column])
        if "parts" in selected:
            integer, fraction = _part_tokens(result[column])
            result[f"{KEY_PREFIX}integer_{column}"] = integer
            result[f"{KEY_PREFIX}fraction_{column}"] = fraction
        if "digits" in selected:
            for digits in (1, 2, 3):
                result[f"{KEY_PREFIX}digit{digits}_{column}"] = _digit_tokens(result[column], digits)
        if "tails" in selected:
            tail, tens = _tail_tokens(result[column])
            result[f"{KEY_PREFIX}tail2_{column}"] = tail
            result[f"{KEY_PREFIX}tens_{column}"] = tens
    if {"score_integers", "score_pair"} & set(selected):
        for column in SCORE_INTEGER_COLUMNS:
            integer, _fraction = _part_tokens(result[column])
            result[f"{KEY_PREFIX}integer_{column}"] = integer
    if "score_pair" in selected:
        left = f"{KEY_PREFIX}integer_{SCORE_INTEGER_COLUMNS[0]}"
        right = f"{KEY_PREFIX}integer_{SCORE_INTEGER_COLUMNS[1]}"
        result[f"{KEY_PREFIX}merchant_velocity_integer"] = result[left].str.cat(result[right], sep="|", na_rep="__MISSING__")
    interactions = []
    if "interactions" in selected:
        interactions.extend([
            (("cvv_retry_count", "auth_method"), f"{KEY_PREFIX}cvv_auth"),
            (("is_foreign_transaction", "auth_method"), f"{KEY_PREFIX}foreign_auth"),
            (("merchant_category", "cvv_retry_count"), f"{KEY_PREFIX}merchant_cvv"),
        ])
    if "interactions_ext" in selected:
        interactions.extend([
            (("merchant_category", "auth_method"), f"{KEY_PREFIX}merchant_auth"),
            (("channel", "is_foreign_transaction"), f"{KEY_PREFIX}channel_foreign"),
            (("cvv_retry_count", "channel"), f"{KEY_PREFIX}cvv_channel"),
            (("merchant_category", "is_foreign_transaction"), f"{KEY_PREFIX}merchant_foreign"),
            (("device_type", "used_vpn"), f"{KEY_PREFIX}device_vpn"),
        ])
    for columns, name in interactions:
        missing = [column for column in columns if column not in result.columns]
        if missing:
            raise ValueError(f"缺少交互编码列: {missing}")
        result[name] = _interaction(result, columns, name)
    for column in [item for item in result.columns if str(item).startswith(KEY_PREFIX)]:
        result[column] = result[column].fillna("__MISSING__").astype("string")
    return result


def _smooth_values(target_smooth) -> list[float]:
    if isinstance(target_smooth, (int, float)):
        return [float(target_smooth)]
    values = [float(value) for value in target_smooth]
    if not values or len(values) != len(set(values)):
        raise ValueError("目标编码平滑强度为空或重复")
    return values


def _smooth_name(value: float, single: bool) -> str:
    if single:
        return "target"
    label = str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")
    return f"target_{label}"


def preprocessor(frame: pd.DataFrame, scale: bool = False, target_encode: bool = False, seed: int = 42, target_smooth: float | list[float] = 20.0) -> ColumnTransformer:
    keys = [column for column in frame.columns if str(column).startswith(KEY_PREFIX)]
    categorical = [column for column in frame.select_dtypes(include=["object", "string", "category"]).columns if column not in keys]
    numeric = [column for column in frame.columns if column not in categorical and column not in keys]
    num_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    transforms = []
    if numeric:
        transforms.append(("numeric", Pipeline(num_steps), numeric))
    if categorical:
        transforms.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
        ]), categorical))
    encoded_columns = ([column for column in frame.columns if column not in keys] if target_encode else []) + keys
    smooths = _smooth_values(target_smooth)
    if encoded_columns:
        # fit_transform(X, y) 使用内部交叉拟合；外层验证只调用 transform。
        # vk_ 键只走目标编码，避免把金额精确值展开成高维独热。
        # 多个平滑强度各出一列，树可以同时使用收缩强弱不同的欺诈率。
        for smooth in smooths:
            transforms.append((
                _smooth_name(smooth, len(smooths) == 1),
                TargetEncoder(target_type="binary", smooth=smooth, cv=5, shuffle=True, random_state=seed),
                encoded_columns,
            ))
    return ColumnTransformer(transforms, sparse_threshold=0, verbose_feature_names_out=bool(encoded_columns))


def cat_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    categorical = result.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for column in categorical:
        result[column] = result[column].fillna("__MISSING__").astype(str)
    for column in result.columns.difference(categorical):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("float32")
    return result, categorical
