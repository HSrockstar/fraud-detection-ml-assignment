"""以 NumPy 实现带 L2 正则的二分类逻辑回归。"""

import numpy as np


def sigmoid(logits):
    return np.exp(-np.logaddexp(0.0, -logits))


def objective(design, labels, weights, strength):
    # design 最后一列为截距，L2 只惩罚其余权重。
    signed_logits = (1.0 - 2.0 * labels) * (design @ weights)
    return float(np.logaddexp(0.0, signed_logits).mean() + 0.5 * strength * (weights[:-1] @ weights[:-1]))


class ScratchLogisticRegression:
    def __init__(self, c=1.0, tol=1e-10, max_iter=100):
        if c <= 0 or tol <= 0 or max_iter < 1:
            raise ValueError("C、收敛容差和迭代上限必须为正")
        self.c = c
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, features, labels):
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if x.ndim != 2 or y.shape != (len(x),) or set(np.unique(y)) != {0.0, 1.0}:
            raise ValueError("需要二维特征和同时包含 0、1 的一维标签")
        design = np.column_stack([x, np.ones(len(x))])
        strength = 1.0 / (self.c * len(y))
        penalty = np.full(design.shape[1], strength)
        penalty[-1] = 0.0
        weights = np.zeros(design.shape[1])
        positive_rate = y.mean()
        weights[-1] = np.log(positive_rate / (1.0 - positive_rate))

        for step in range(self.max_iter + 1):
            logits = design @ weights
            probability = sigmoid(logits)
            gradient = design.T @ (probability - y) / len(y) + penalty * weights
            gradient_norm = np.linalg.norm(gradient, ord=np.inf)
            if gradient_norm <= self.tol:
                self.coef_ = weights[:-1].copy()
                self.intercept_ = float(weights[-1])
                self.n_iter_ = step
                self.gradient_norm_ = float(gradient_norm)
                return self
            if step == self.max_iter:
                break

            curvature = probability * sigmoid(-logits)
            hessian = design.T @ (curvature[:, None] * design) / len(y) + np.diag(penalty)
            direction = np.linalg.solve(hessian, gradient)
            loss = objective(design, y, weights, strength)
            slope = gradient @ direction
            # 回溯线搜索保证病态输入上的完整牛顿步也经目标函数检验。
            scale = 1.0
            while scale >= 2**-20:
                candidate = weights - scale * direction
                if objective(design, y, candidate, strength) <= loss - 1e-4 * scale * slope:
                    weights = candidate
                    break
                scale *= 0.5
            else:
                raise RuntimeError("牛顿法未找到下降步长")
        raise RuntimeError(f"手写逻辑回归未收敛，梯度无穷范数为 {gradient_norm:.3g}")

    def predict_proba(self, features):
        positive = sigmoid(np.asarray(features, dtype=np.float64) @ self.coef_ + self.intercept_)
        return np.column_stack([1.0 - positive, positive])
