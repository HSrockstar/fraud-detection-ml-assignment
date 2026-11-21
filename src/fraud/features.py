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


def preprocessor(frame: pd.DataFrame, scale: bool = False, target_encode: bool = False, seed: int = 42) -> ColumnTransformer:
    categorical = frame.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = [c for c in frame.columns if c not in categorical]
    num_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    transforms = [
        ("numeric", Pipeline(num_steps), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
        ]), categorical),
    ]
    if target_encode:
        # fit_transform(X, y) 使用内部交叉拟合；外层验证只调用 transform。
        transforms.append(("target", TargetEncoder(target_type="binary", smooth=20.0, cv=5, shuffle=True, random_state=seed), frame.columns.tolist()))
    return ColumnTransformer(transforms, sparse_threshold=0, verbose_feature_names_out=target_encode)


def cat_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    categorical = result.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for column in categorical:
        result[column] = result[column].fillna("__MISSING__").astype(str)
    for column in result.columns.difference(categorical):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("float32")
    return result, categorical
