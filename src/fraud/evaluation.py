from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                            f1_score, log_loss, precision_recall_curve,
                            precision_score, recall_score, roc_auc_score)


def f1_threshold(y, probabilities) -> float:
    """阈值只用开发集 OOF 或内部验证预测确定。"""
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    score = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[np.argmax(score)])


def metrics(y, probabilities, threshold: float = 0.5) -> dict:
    probabilities = np.asarray(probabilities, dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("预测概率必须有限且位于 [0, 1]")
    prediction = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
    }
